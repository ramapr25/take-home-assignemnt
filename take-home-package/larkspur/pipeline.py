"""End-to-end pipeline: notes -> rule fields -> LLM fields -> predictions -> evaluation."""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from concurrent.futures import ThreadPoolExecutor

import anthropic

from larkspur import config
from larkspur.evaluation import evaluate
from larkspur.extractors import extract_rule_baseline, extract_rule_fields
from larkspur.jsonl import read_jsonl, write_jsonl
from larkspur.llm_extractor import LLMExtractor
from larkspur.schema import FIELDS

log = logging.getLogger("larkspur")


def _as_prediction(note_id: str, fields: dict) -> dict:
    """One output row, with columns in schema order."""
    return {"note_id": note_id, **{f: fields[f] for f in FIELDS}}


def run_rules(notes: list[dict]) -> dict[str, dict]:
    """All six fields from rules alone, keyed by note_id."""
    out = {}
    for note in notes:
        fields = extract_rule_fields(note)
        fields.update(extract_rule_baseline(note, fields["model_number"]))
        out[note["note_id"]] = fields
    return out


def run_llm(notes: list[dict], model: str, fallback_model: str | None,
            workers: int) -> tuple[dict[str, dict | None], dict]:
    """LLM fields for every note (None where every model failed), plus usage stats."""
    extractor = LLMExtractor(model=model, fallback_model=fallback_model)

    def work(note: dict) -> tuple[str, dict | None, str | None]:
        try:
            return note["note_id"], extractor.extract(note["text"]), None
        except anthropic.AuthenticationError:
            raise  # a bad key fails every note; stop instead of falling back 500 times
        except Exception as exc:  # one bad note must not stop the batch
            return note["note_id"], None, repr(exc)

    results: dict[str, dict | None] = {}
    failures: list[dict] = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for done, (note_id, answer, error) in enumerate(pool.map(work, notes), start=1):
            results[note_id] = answer
            if error:
                failures.append({"note_id": note_id, "error": error})
                log.warning("%s: all models failed, using rule baseline (%s)", note_id, error)
            if done % 50 == 0 or done == len(notes):
                log.info("LLM progress: %d/%d notes", done, len(notes))

    stats = extractor.summary(len(notes))
    stats["answered_by"]["rules"] = len(failures)
    stats["failures"] = failures
    return results, stats


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Extract structured fields from Larkspur repair notes.",
    )
    parser.add_argument("--dry-run", action="store_true",
                        help="skip the LLM; categorical fields come from the rule baseline")
    parser.add_argument("--model", default=None,
                        help=f"Claude model id (default: $LARKSPUR_MODEL or {config.DEFAULT_MODEL})")
    parser.add_argument("--fallback-model", default=None,
                        help="model to try when the primary fails, or 'none' "
                             f"(default: $LARKSPUR_FALLBACK_MODEL or {config.DEFAULT_FALLBACK_MODEL})")
    parser.add_argument("--workers", type=int, default=config.DEFAULT_WORKERS,
                        help="concurrent API calls (default: %(default)s)")
    parser.add_argument("--only-labeled", action="store_true",
                        help="process only the labeled notes (cheap prompt iteration)")
    parser.add_argument("--limit", type=int, default=None, help="process only the first N notes")
    parser.add_argument("--no-eval", action="store_true", help="skip the evaluation report")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)  # one line per API call is noise
    config.load_env()

    notes = read_jsonl(config.NOTES_PATH)
    if args.only_labeled:
        labeled_ids = {row["note_id"] for row in read_jsonl(config.LABELS_PATH)}
        notes = [n for n in notes if n["note_id"] in labeled_ids]
    if args.limit:
        notes = notes[: args.limit]
    is_subset = bool(args.only_labeled or args.limit)
    log.info("loaded %d notes", len(notes))

    # 1. Rules: always run. Their categorical guesses are the baseline / fallback.
    rules = run_rules(notes)
    write_jsonl(config.RULE_BASELINE_PATH, (_as_prediction(n["note_id"], rules[n["note_id"]]) for n in notes))

    if args.dry_run:
        out_path = config.DRY_RUN_PREDICTIONS_PATH
        write_jsonl(out_path, (_as_prediction(n["note_id"], rules[n["note_id"]]) for n in notes))
        log.info("dry run: wrote %d rows to %s", len(notes), config.display_path(out_path))
    else:
        # 2. LLM for the categorical fields.
        if not os.environ.get("ANTHROPIC_API_KEY"):
            log.error("ANTHROPIC_API_KEY is not set. Add it to .env (see how_to_run.md) or use --dry-run.")
            return 1
        model = args.model or config.model_from_env()
        if args.fallback_model:
            fallback = None if args.fallback_model.lower() == "none" else args.fallback_model
        else:
            fallback = config.fallback_model_from_env()
        log.info("calling %s (fallback: %s, then rules) with %d workers", model, fallback, args.workers)
        try:
            llm_answers, stats = run_llm(notes, model, fallback, args.workers)
        except anthropic.AuthenticationError:
            log.error("The API key was rejected. Check ANTHROPIC_API_KEY, or use --dry-run.")
            return 1

        # 3. Merge: LLM wins for its fields; rules fill in notes where every model failed.
        rows = []
        for note in notes:
            fields = dict(rules[note["note_id"]])
            fields.update(llm_answers[note["note_id"]] or {})
            rows.append(_as_prediction(note["note_id"], fields))

        out_path = config.PARTIAL_PREDICTIONS_PATH if is_subset else config.PREDICTIONS_PATH
        write_jsonl(out_path, rows)
        config.RUN_STATS_PATH.write_text(json.dumps(stats, indent=2), encoding="utf-8")
        log.info("wrote %d rows to %s (%d LLM failures)", len(rows),
                 config.display_path(out_path), len(stats["failures"]))
        print(json.dumps({k: v for k, v in stats.items() if k != "failures"}, indent=2))

    # 4. Evaluate. Comparing the dry run to the baseline would compare it with itself.
    if not args.no_eval:
        evaluate(out_path, baseline_path=None if args.dry_run else config.RULE_BASELINE_PATH)
    return 0


if __name__ == "__main__":
    sys.exit(main())
