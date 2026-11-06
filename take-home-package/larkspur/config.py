"""Project paths, model settings, prices and .env loading, kept in one place."""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

# --- Paths -----------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"
NOTES_PATH = DATA_DIR / "notes.jsonl"
LABELS_PATH = DATA_DIR / "labeled_sample.jsonl"
SCHEMA_PATH = DATA_DIR / "schema.json"

# The deliverable sits at the repo root; diagnostics go to outputs/.
PREDICTIONS_PATH = PROJECT_ROOT / "predictions.jsonl"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
DRY_RUN_PREDICTIONS_PATH = OUTPUT_DIR / "predictions_dryrun.jsonl"
PARTIAL_PREDICTIONS_PATH = OUTPUT_DIR / "predictions_partial.jsonl"
RULE_BASELINE_PATH = OUTPUT_DIR / "predictions_rules.jsonl"
ERRORS_PATH = OUTPUT_DIR / "errors.jsonl"
DISAGREEMENTS_PATH = OUTPUT_DIR / "disagreements.jsonl"
RUN_STATS_PATH = OUTPUT_DIR / "run_stats.json"

# --- Model -----------------------------------------------------------------

DEFAULT_MODEL = "claude-haiku-4-5"
# Tried when the primary model fails on a note or is unavailable to this key.
# If both models fail, the note falls back to the rule baseline.
DEFAULT_FALLBACK_MODEL = "claude-haiku-5-5"
DEFAULT_WORKERS = 8

# USD per million tokens (input, output), standard Claude API rates.
# The Message Batches API is 50% of these.
PRICING_PER_MTOK: dict[str, tuple[float, float]] = {
    "claude-haiku-4-5": (1.00, 5.00),
    "claude-haiku-5-5": (0.10, 0.50),
}


def load_env() -> None:
    """Load `.env` from the project root, if present. Shell variables take precedence."""
    load_dotenv(PROJECT_ROOT / ".env", override=False)


def display_path(path: Path) -> str:
    """Path relative to the project root for log messages, absolute if outside it."""
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def model_from_env() -> str:
    """Model id from LARKSPUR_MODEL, falling back to the default."""
    return os.environ.get("LARKSPUR_MODEL") or DEFAULT_MODEL


def fallback_model_from_env() -> str | None:
    """Fallback model id from LARKSPUR_FALLBACK_MODEL; set it to "none" to disable."""
    value = os.environ.get("LARKSPUR_FALLBACK_MODEL") or DEFAULT_FALLBACK_MODEL
    return None if value.lower() == "none" else value
