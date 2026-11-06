"""LLM extraction of appliance_type, problem_category and part_replaced.

One Messages API call per note. The response is constrained with structured
outputs (`output_config.format` = a JSON schema built from the enums in
schema.json), so the model can only return valid labels. Output is still
validated, and bad responses are retried.

The Anthropic SDK reads the API key from the ANTHROPIC_API_KEY environment
variable (which config.load_env() can populate from a .env file).
"""
from __future__ import annotations

import json
import logging
import threading
import time
from collections import Counter
from dataclasses import asdict, dataclass

import anthropic

from larkspur.config import DEFAULT_MODEL, PRICING_PER_MTOK
from larkspur.schema import APPLIANCE_TYPES, LLM_FIELDS, PARTS, PROBLEM_CATEGORIES

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Prompt and output schema
# ---------------------------------------------------------------------------

OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "appliance_type": {"type": "string", "enum": APPLIANCE_TYPES},
        "problem_category": {"type": "string", "enum": PROBLEM_CATEGORIES},
        "part_replaced": {"type": "string", "enum": PARTS},
    },
    "required": LLM_FIELDS,
    "additionalProperties": False,
}

SYSTEM_PROMPT = """\
You extract structured fields from appliance-repair technician notes for Larkspur Appliance Service. Each note describes one service visit. Return only the JSON object the schema asks for.

## Fields

appliance_type: the appliance that was serviced on this visit.
- refrigerator: fridge, refrig, French-door fridge, side-by-side
- oven: oven, wall oven, range, stove
- microwave: microwave, OTR micro, over-the-range microwave
- dishwasher: dishwasher, dish washer, DW
- washer: washer, washing machine, front loader, top-load washer
- dryer: dryer, gas dryer, electric dryer
- water_heater: water heater, w/h, hot water tank
Ignore any second appliance in a sentence like "Customer also asked me to look at the X (...), checked it, no issues found." That appliance was not the one serviced.

problem_category: the customer's reported symptom (what the customer complained about), not the technician's diagnosis and not the part that was replaced.
- door_or_latch: door seal torn, latch broken, door won't latch, door won't close right
- leaking: leaking, drip from the bottom, water on the floor, puddle under unit
- noise: squealing, grinding sound, rattling when running, loud noise
- not_cooling: not cooling, warm inside, freezer ok but fridge warm, fresh food side at 55F
- not_heating: no heat, not heating, takes forever to heat, runs but stays cold
- wont_start: won't start, won't turn on, nothing happens when you press start, dead / no power
- error_code: ONLY when an error code is the only thing reported, e.g. "showing F3 on display, otherwise runs". If the customer reports a symptom and the display also shows a code ("leaking, display also shows OE"), use the symptom's category, not error_code.

part_replaced: the part actually installed on this visit, or "none".
- "Replaced X" / "Installed new X" / "Actual fault was the X, replaced it" means X was installed.
- Abbreviations: "board" = control board, "tstat" = thermostat, "inlet valve" = water inlet valve.
- "Customer thought it was X. Tested X, it is fine. Actual fault was Y, replaced it." means the part is Y, not X.
- "Pulled LPN-1234 from stock. ... Installed new <part> (part LPN-1234)" means the part is <part>. Part/stock numbers are never the answer.
- A part that was only diagnosed or ordered ("Diagnosed bad X; X ordered, not in van") is "none".
- "No parts needed", "no parts used", "Cleaned and reset, no parts" mean "none".
- A part that was replaced on this visit still counts even if the note also says something is backordered or a return visit is booked."""


# ---------------------------------------------------------------------------
# Usage / cost tracking
# ---------------------------------------------------------------------------

@dataclass
class Usage:
    """Token totals across calls. Thread-safe via the owning extractor's lock."""

    calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0

    def summary(self, model: str, notes: int) -> dict:
        """Totals, averages and cost (if the model's price is known)."""
        out: dict = {"model": model, "notes": notes, **asdict(self)}
        if not self.calls or model not in PRICING_PER_MTOK:
            return out
        price_in, price_out = PRICING_PER_MTOK[model]
        cost = self.input_tokens / 1e6 * price_in + self.output_tokens / 1e6 * price_out
        out.update(
            price_in_per_mtok=price_in,
            price_out_per_mtok=price_out,
            avg_input_tokens=round(self.input_tokens / self.calls, 1),
            avg_output_tokens=round(self.output_tokens / self.calls, 1),
            cost_this_run_usd=round(cost, 4),
            cost_per_1000_notes_usd=round(cost / notes * 1000, 4) if notes else None,
        )
        return out


# ---------------------------------------------------------------------------
# Extractor
# ---------------------------------------------------------------------------

class ExtractionError(RuntimeError):
    """No model returned a valid answer for the note."""


# The key itself is bad: every model will fail the same way, so stop the run.
_KEY_ERRORS = (anthropic.AuthenticationError,)
# This model can't be used with this key (unknown id, no access). Skip it for
# the rest of the run and go straight to the fallback.
_MODEL_UNAVAILABLE = (anthropic.NotFoundError, anthropic.PermissionDeniedError)
# This request was rejected; retrying the same model won't help, the fallback might.
_REQUEST_REJECTED = (anthropic.BadRequestError,)


class LLMExtractor:
    """Extracts the categorical fields from one note per call. Safe to share across threads.

    Fallback chain per note: primary model -> fallback model -> caller's rule
    baseline (the caller handles that last step when this raises).
    """

    def __init__(self, model: str = DEFAULT_MODEL, fallback_model: str | None = None,
                 max_attempts: int = 3) -> None:
        # The SDK already retries 429 / 5xx / connection errors (max_retries);
        # max_attempts additionally covers malformed or invalid responses.
        self.client = anthropic.Anthropic(max_retries=4)
        self.models = [model] + ([fallback_model] if fallback_model and fallback_model != model else [])
        self.max_attempts = max_attempts
        self.usage = {m: Usage() for m in self.models}
        self.answered_by: Counter[str] = Counter()
        self._unavailable: set[str] = set()
        self._lock = threading.Lock()

    def _request_params(self, model: str, text: str) -> dict:
        params = {
            "model": model,
            "max_tokens": 200,  # the answer is ~30 tokens of JSON
            "system": SYSTEM_PROMPT,
            "output_config": {"format": {"type": "json_schema", "schema": OUTPUT_SCHEMA}},
            "messages": [{"role": "user", "content": f"<note>\n{text}\n</note>"}],
        }
        # Haiku 4.5 accepts temperature on the API version that anthropic<1.0
        # sends (2023-06-01). Newer models such as Haiku 5.5 reject sampling
        # parameters and think by default; thinking tokens would eat the
        # 200-token budget and cost money, so turn it off at low effort.
        if model.startswith("claude-haiku-4-5"):
            params["temperature"] = 0
        else:
            params["thinking"] = {"type": "disabled"}
            params["output_config"]["effort"] = "low"
        return params

    def _record_usage(self, model: str, response) -> None:
        with self._lock:
            usage = self.usage[model]
            usage.calls += 1
            usage.input_tokens += response.usage.input_tokens
            usage.output_tokens += response.usage.output_tokens

    def _extract_with(self, model: str, text: str) -> dict:
        """One model, with retries for transient failures and invalid answers."""
        last_error: Exception | None = None
        for attempt in range(1, self.max_attempts + 1):
            try:
                response = self.client.messages.create(**self._request_params(model, text))
                self._record_usage(model, response)
                if response.stop_reason == "refusal":
                    raise ValueError("model refused the request")
                raw = next((b.text for b in response.content if b.type == "text"), None)
                if raw is None:
                    raise ValueError(f"response has no text block (stop_reason={response.stop_reason})")
                return validate_labels(json.loads(raw))
            except (*_KEY_ERRORS, *_MODEL_UNAVAILABLE, *_REQUEST_REJECTED):
                raise
            except (anthropic.APIError, ValueError) as exc:  # JSONDecodeError is a ValueError
                last_error = exc
                log.warning("%s attempt %d/%d failed: %s", model, attempt, self.max_attempts, exc)
                if attempt < self.max_attempts:
                    time.sleep(1.5 * attempt)
        raise ExtractionError(f"{model}: no valid answer after {self.max_attempts} attempts: {last_error}")

    def extract(self, text: str) -> dict:
        """Return {appliance_type, problem_category, part_replaced} for one note."""
        errors = []
        for model in self.models:
            if model in self._unavailable:
                continue
            try:
                answer = self._extract_with(model, text)
            except _KEY_ERRORS:
                raise
            except _MODEL_UNAVAILABLE as exc:
                with self._lock:
                    first_time = model not in self._unavailable
                    self._unavailable.add(model)
                if first_time:
                    log.warning("%s is unavailable (%s); using the fallback for the rest of the run",
                                model, type(exc).__name__)
                errors.append(f"{model}: {exc!r}")
                continue
            except (*_REQUEST_REJECTED, ExtractionError) as exc:
                errors.append(f"{model}: {exc!r}")
                continue
            with self._lock:
                self.answered_by[model] += 1
            return answer
        raise ExtractionError("; ".join(errors) or "no usable model")

    def summary(self, notes: int) -> dict:
        """Run totals plus a per-model breakdown (models price differently)."""
        per_model = {m: u.summary(m, notes) for m, u in self.usage.items() if u.calls}
        cost = sum(s.get("cost_this_run_usd", 0) for s in per_model.values())
        return {
            "model": self.models[0],
            "fallback_model": self.models[1] if len(self.models) > 1 else None,
            "notes": notes,
            "answered_by": dict(self.answered_by),
            "calls": sum(u.calls for u in self.usage.values()),
            "input_tokens": sum(u.input_tokens for u in self.usage.values()),
            "output_tokens": sum(u.output_tokens for u in self.usage.values()),
            "cost_this_run_usd": round(cost, 4),
            "cost_per_1000_notes_usd": round(cost / notes * 1000, 4) if notes else None,
            "per_model": per_model,
        }


def validate_labels(answer: dict) -> dict:
    """Check every label is in its enum; raise ValueError otherwise."""
    allowed = {
        "appliance_type": APPLIANCE_TYPES,
        "problem_category": PROBLEM_CATEGORIES,
        "part_replaced": PARTS,
    }
    out = {}
    for field, labels in allowed.items():
        value = answer.get(field)
        if value not in labels:
            raise ValueError(f"invalid {field}: {value!r}")
        out[field] = value
    return out
