"""Rule-based extractors. No LLM, no network.

Primary rule fields
    model_number, visit_date, follow_up_needed

Rule baseline for the LLM fields
    appliance_type, problem_category, part_replaced

The LLM is the primary extractor for the categorical fields. The baseline here
serves three purposes: it powers `--dry-run`, it is the fallback when an LLM
call fails, and it is a cross-check (evaluation reports every note where the
LLM and the rules disagree, which covers all 500 notes, not just the 50 labels).
"""
from __future__ import annotations

import re
from datetime import date, datetime, timedelta

from dateutil import parser as dateparser

from larkspur.schema import PARTS

DEFAULT_YEAR = 2026  # every visit in the dataset is in 2026

# ---------------------------------------------------------------------------
# Shared cleanup
# ---------------------------------------------------------------------------

# A second appliance the tech only looked at, e.g.
#   "Customer also asked me to look at the dish washer (LDW-1695A), checked it, no issues found."
# Its name and model number must not leak into the serviced unit's fields.
_SECONDARY_APPLIANCE = re.compile(
    r"Customer also asked me to look at the [^()]*\([^)]*\)[^.]*\.?", re.IGNORECASE
)

# Part and stock numbers. LPN-#### also fits the model-number pattern, so these
# are removed before the model regex runs.
_PART_NUMBER = re.compile(r"\b(?:LPN|SKU|PN|WP)-\d+[A-Z]?\b")
_PART_CONTEXT = re.compile(
    r"part\s*#\s*\S+"            # "Used part #WP-49668 from van stock"
    r"|\(part [^)]*\)"           # "Installed new heating element (part LPN-5648)"
    r"|Pulled \S+ from stock",   # "Pulled LPN-5648 from stock"
    re.IGNORECASE,
)


def strip_secondary_appliance(text: str) -> str:
    """Remove clauses about an appliance that was checked but not serviced."""
    return _SECONDARY_APPLIANCE.sub(" ", text)


# ---------------------------------------------------------------------------
# model_number
# ---------------------------------------------------------------------------

_MODEL_NUMBER = re.compile(r"\b[A-Z]{3}-\d{4}[A-Z]?\b")
_NO_MODEL = re.compile(
    r"model plate unreadable"
    r"|could not find model tag"
    r"|no model\s*#?\s*on unit"
    r"|model (?:tag|plate|number) (?:missing|illegible|unreadable)",
    re.IGNORECASE,
)


def extract_model_number(text: str) -> str | None:
    """The serviced unit's model number, or None if unreadable / absent."""
    text = strip_secondary_appliance(text)
    if _NO_MODEL.search(text):
        return None
    text = _PART_CONTEXT.sub(" ", text)
    text = _PART_NUMBER.sub(" ", text)
    match = _MODEL_NUMBER.search(text)
    return match.group(0) if match else None


# ---------------------------------------------------------------------------
# visit_date
# ---------------------------------------------------------------------------

_MONTH = r"(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)[a-z]*\.?"
_VISIT = r"\bVisit(?:ed)?\s+"

_ISO_DATE = re.compile(_VISIT + r"(\d{4})-(\d{1,2})-(\d{1,2})\b", re.IGNORECASE)        # 2026-03-22
_US_DATE = re.compile(_VISIT + r"(\d{1,2})/(\d{1,2})(?:/(\d{2,4}))?\b", re.IGNORECASE)  # 03/22[/2026]
_TEXT_DATES = [
    re.compile(_VISIT + rf"({_MONTH}\s+\d{{1,2}}(?:,?\s+\d{{4}})?)", re.IGNORECASE),     # Mar 22[, 2026]
    re.compile(_VISIT + rf"(\d{{1,2}}\s+{_MONTH}(?:,?\s+\d{{4}})?)", re.IGNORECASE),     # 22 Mar [2026]
]
_RELATIVE_DATE = re.compile(_VISIT + r"(today|yesterday)\b", re.IGNORECASE)


def _submitted_date(submitted_at: str) -> date:
    return datetime.fromisoformat(submitted_at.replace("Z", "+00:00")).date()


def extract_visit_date(text: str, submitted_at: str) -> str | None:
    """Visit date as YYYY-MM-DD.

    "today" / "yesterday" resolve against the UTC date of `submitted_at`. All
    notes were submitted 15:00-23:59 UTC, which is the same calendar day in
    every US time zone, so the UTC date is safe for this data.
    """
    if match := _RELATIVE_DATE.search(text):
        visit = _submitted_date(submitted_at)
        if match.group(1).lower() == "yesterday":
            visit -= timedelta(days=1)
        return visit.isoformat()

    if match := _ISO_DATE.search(text):
        year, month, day = (int(g) for g in match.groups())
        return date(year, month, day).isoformat()

    if match := _US_DATE.search(text):
        month, day, year = match.groups()
        year = int(year) if year else DEFAULT_YEAR
        if year < 100:
            year += 2000
        return date(year, int(month), int(day)).isoformat()

    for pattern in _TEXT_DATES:
        if match := pattern.search(text):
            parsed = dateparser.parse(match.group(1), default=datetime(DEFAULT_YEAR, 1, 1))
            return parsed.date().isoformat()

    return None


# ---------------------------------------------------------------------------
# follow_up_needed
# ---------------------------------------------------------------------------

# True only when a return visit is scheduled or a part is on order.
_FOLLOW_UP = re.compile(
    r"will return to install"
    r"|ordered, not in van"
    r"|back-?ordered"
    r"|rebook when it arrives"
    r"|on order\b"
    r"|scheduled (?:a )?second visit"
    r"|need a second tech"
    r"|booked for next week"
    r"|will be back",
    re.IGNORECASE,
)
# Phrases that mean no follow-up ("call us if it happens again", "fixed on first
# visit", "no follow-up required", "cx declined a return visit") need no rule:
# anything without a follow-up phrase is False.


def extract_follow_up(text: str) -> bool:
    return bool(_FOLLOW_UP.search(strip_secondary_appliance(text)))


# ---------------------------------------------------------------------------
# Rule baseline: appliance_type, problem_category, part_replaced
# ---------------------------------------------------------------------------

# The model-number prefix identifies the appliance line. This agrees with the
# appliance word in the text on all 500 notes.
_MODEL_PREFIX_TO_APPLIANCE = {
    "LRF": "refrigerator",
    "LOV": "oven",
    "LMW": "microwave",
    "LDW": "dishwasher",
    "LWM": "washer",
    "LDR": "dryer",
    "LWH": "water_heater",
}

# Checked in order: "dish washer" before "washer", "water heater" before "heat".
_APPLIANCE_WORDS = [
    (r"dish ?washer|\bDW\b", "dishwasher"),
    (r"water heater|hot water tank|\bw/h\b", "water_heater"),
    (r"microwave|\bOTR micro\b|\bmicro\b", "microwave"),
    (r"refrigerator|refrig|fridge|side-by-side", "refrigerator"),
    (r"washing machine|front loader|top-load washer|\bwasher\b", "washer"),
    (r"dryer", "dryer"),
    (r"oven|range|stove", "oven"),
]

# Checked in order. error_code comes first because it only applies to the
# "code is the only symptom" phrasing; "<symptom>, display also shows X" falls
# through to the symptom's category.
_PROBLEM_PHRASES = [
    (r"showing \w+ on display, otherwise runs", "error_code"),
    (r"door seal torn|latch broken|door won't latch|door won't close", "door_or_latch"),
    (r"leak|drip|water on the floor|puddle", "leaking"),
    (r"squeal|grind|rattl|noise", "noise"),
    (r"not cooling|warm inside|fridge warm|fresh food side at \d+F", "not_cooling"),
    (r"no heat|not heating|takes forever to heat|stays cold", "not_heating"),
    (r"won't start|won't turn on|nothing happens when you press start|dead, no power", "wont_start"),
]

_PART_ABBREVIATIONS = {"board": "control board", "tstat": "thermostat", "inlet valve": "water inlet valve"}
_REAL_PARTS = [p for p in PARTS if p != "none"]
# Longest first, so "control board" wins over the abbreviation "board".
_PART_ALTERNATION = "|".join(
    re.escape(p) for p in sorted(_REAL_PARTS + list(_PART_ABBREVIATIONS), key=len, reverse=True)
)
# Phrasings that mean a part was installed on this visit, most specific first.
# "Customer thought it was X. Tested X, it is fine." never matches these, and
# "Diagnosed bad X; X ordered" doesn't either, so ordered parts stay "none".
_INSTALLED_PART = [
    re.compile(rf"Actual fault was the ({_PART_ALTERNATION}), replaced it", re.IGNORECASE),
    re.compile(rf"Installed new ({_PART_ALTERNATION})\b", re.IGNORECASE),
    re.compile(rf"\bReplaced ({_PART_ALTERNATION})\b", re.IGNORECASE),
]

_COMPLAINT_SENTENCES = [
    re.compile(r"Complaint:[^.]*", re.IGNORECASE),                       # Complaint: no heat (oven).
    re.compile(r"Cust reports[^.]*", re.IGNORECASE),                     # Cust reports DW rattling.
    re.compile(r"[^.]* - [^.]* per c(?:ustomer|x)", re.IGNORECASE),      # Dryer - won't start per customer.
]


def _complaint_sentence(text: str) -> str:
    """The sentence that carries the customer's complaint (whole note if none)."""
    text = strip_secondary_appliance(text)
    for pattern in _COMPLAINT_SENTENCES:
        if match := pattern.search(text):
            return match.group(0)
    return text


def rule_appliance_type(text: str, model_number: str | None) -> str | None:
    if model_number and (appliance := _MODEL_PREFIX_TO_APPLIANCE.get(model_number[:3])):
        return appliance
    complaint = _complaint_sentence(text)
    for pattern, label in _APPLIANCE_WORDS:
        if re.search(pattern, complaint, re.IGNORECASE):
            return label
    return None


def rule_problem_category(text: str) -> str | None:
    complaint = _complaint_sentence(text)
    for pattern, label in _PROBLEM_PHRASES:
        if re.search(pattern, complaint, re.IGNORECASE):
            return label
    return None


def rule_part_replaced(text: str) -> str:
    text = strip_secondary_appliance(text)
    for pattern in _INSTALLED_PART:
        if match := pattern.search(text):
            part = match.group(1).lower()
            return _PART_ABBREVIATIONS.get(part, part)
    return "none"


# ---------------------------------------------------------------------------
# Entry points
# ---------------------------------------------------------------------------

def extract_rule_fields(note: dict) -> dict:
    """The three fields that rules own in the hybrid pipeline."""
    text = note["text"]
    return {
        "model_number": extract_model_number(text),
        "visit_date": extract_visit_date(text, note["submitted_at"]),
        "follow_up_needed": extract_follow_up(text),
    }


def extract_rule_baseline(note: dict, model_number: str | None) -> dict:
    """Rule-based guesses for the three fields the LLM owns."""
    text = note["text"]
    return {
        "appliance_type": rule_appliance_type(text, model_number),
        "problem_category": rule_problem_category(text),
        "part_replaced": rule_part_replaced(text),
    }
