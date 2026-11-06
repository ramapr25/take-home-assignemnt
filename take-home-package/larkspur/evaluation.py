"""Score predictions against the labeled sample.

Produces:
  * per-field accuracy on the labeled notes (printed)
  * confusion matrices for the three categorical fields (printed)
  * every field mismatch, with the note text  -> outputs/errors.jsonl
  * optionally, where the predictions disagree with the rule baseline across
    all notes                                 -> outputs/disagreements.jsonl
"""
from __future__ import annotations

from collections import Counter
from pathlib import Path

from larkspur import config
from larkspur.jsonl import read_jsonl, write_jsonl
from larkspur.schema import FIELDS, LLM_FIELDS


def _print_confusion_matrix(field: str, pairs: list[tuple]) -> None:
    """pairs are (predicted, actual). Rows = actual, columns = predicted."""
    labels = sorted({p for p, _ in pairs} | {a for _, a in pairs}, key=str)
    counts = Counter(pairs)
    row_w = max(len(str(label)) for label in labels) + 2
    col_w = 10

    print(f"\nConfusion matrix: {field}  (rows = actual, columns = predicted)")
    print(" " * row_w + "".join(f"{str(label)[:col_w - 1]:>{col_w}}" for label in labels))
    for actual in labels:
        cells = "".join(f"{counts[(pred, actual)] or '.':>{col_w}}" for pred in labels)
        print(f"{str(actual):<{row_w}}{cells}")


def evaluate(
    predictions_path: Path,
    *,
    labels_path: Path = config.LABELS_PATH,
    notes_path: Path = config.NOTES_PATH,
    errors_path: Path = config.ERRORS_PATH,
    baseline_path: Path | None = None,
    disagreements_path: Path = config.DISAGREEMENTS_PATH,
) -> dict[str, float]:
    """Print the evaluation report and return per-field accuracy."""
    gold = {row["note_id"]: row for row in read_jsonl(labels_path)}
    preds = {row["note_id"]: row for row in read_jsonl(predictions_path)}
    texts = {row["note_id"]: row["text"] for row in read_jsonl(notes_path)}

    ids = [note_id for note_id in gold if note_id in preds]
    if not ids:
        print(f"No labeled notes found in {predictions_path}; nothing to score.")
        return {}

    # --- Per-field accuracy -------------------------------------------------
    print(f"\n=== {predictions_path.name}: accuracy on {len(ids)} labeled notes ===")
    print(f"{'field':<22}{'correct':>9}{'accuracy':>10}")
    accuracy: dict[str, float] = {}
    errors = []
    for field in FIELDS:
        correct = 0
        for note_id in ids:
            predicted, actual = preds[note_id].get(field), gold[note_id][field]
            if predicted == actual:
                correct += 1
            else:
                errors.append({"note_id": note_id, "field": field, "predicted": predicted,
                               "actual": actual, "text": texts.get(note_id)})
        accuracy[field] = correct / len(ids)
        print(f"{field:<22}{correct:>5}/{len(ids):<3}{accuracy[field]:>10.1%}")

    all_correct = sum(all(preds[i].get(f) == gold[i][f] for f in FIELDS) for i in ids)
    print(f"{'all six fields':<22}{all_correct:>5}/{len(ids):<3}{all_correct / len(ids):>10.1%}")

    # --- Confusion matrices -------------------------------------------------
    for field in LLM_FIELDS:
        _print_confusion_matrix(field, [(preds[i].get(field), gold[i][field]) for i in ids])

    # --- Mismatch dump ------------------------------------------------------
    write_jsonl(errors_path, errors)
    print(f"\n{len(errors)} field mismatches written to {config.display_path(errors_path)}")
    for e in errors[:15]:
        print(f"  {e['note_id']}  {e['field']}: predicted={e['predicted']!r}  actual={e['actual']!r}")

    # --- Agreement with the rule baseline (all notes, not just labeled) -----
    if baseline_path and baseline_path.exists():
        _report_disagreements(preds, baseline_path, texts, disagreements_path)

    return accuracy


def _report_disagreements(preds: dict, baseline_path: Path, texts: dict, out_path: Path) -> None:
    baseline = {row["note_id"]: row for row in read_jsonl(baseline_path)}
    common = [note_id for note_id in preds if note_id in baseline]

    print(f"\n=== Agreement with the rule baseline on all {len(common)} notes ===")
    disagreements = []
    for field in LLM_FIELDS:
        differ = [i for i in common if preds[i].get(field) != baseline[i].get(field)]
        print(f"{field:<22}{len(common) - len(differ):>5}/{len(common)} agree")
        disagreements += [
            {"note_id": i, "field": field, "predicted": preds[i].get(field),
             "rules": baseline[i].get(field), "text": texts.get(i)}
            for i in differ
        ]
    write_jsonl(out_path, disagreements)
    print(f"{len(disagreements)} disagreements written to {config.display_path(out_path)}")
