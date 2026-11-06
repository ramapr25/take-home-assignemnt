# Tech stack

## Language and runtime

| Item | Version | Used for |
|---|---|---|
| Python | 3.10 or newer | The whole pipeline. |

## Python libraries

These are in `requirements.txt`.

| Library | Version | Used for |
|---|---|---|
| `anthropic` | at least 0.82 and below 1.0 | The official Anthropic SDK, used to call Claude. It also provides automatic retries, typed errors and token-usage reporting. |
| `python-dateutil` | 2.8 or newer | Parsing written-out dates such as "Mar 22" and "22 March 2026". |
| `python-dotenv` | 1.0 or newer | Loading `ANTHROPIC_API_KEY` and the model settings from `.env`. |

The `anthropic` SDK stays below 1.0 for a reason. Version 1.x sends a newer API version, and that version rejects `temperature`. The pipeline needs `temperature=0` on Haiku 4.5 for repeatable answers.

Everything else comes from the Python standard library:

| Module | Used for |
|---|---|
| `re` | All the rule-based extractors. |
| `json` | Reading and writing JSON Lines files and `schema.json`. |
| `concurrent.futures` | Sending API calls in parallel; there are 8 workers by default. |
| `threading` | Keeping the shared token counters thread-safe. |
| `argparse` | The command line. |
| `logging` | Progress and warning messages. |
| `dataclasses` | The token-usage record. |
| `collections` | Counting which model or the rules answered each note. |
| `pathlib` | File paths. |
| `datetime` | Resolving "today" and "yesterday". |

## AI models (Anthropic Claude API)

| Role | Model | Price per million tokens (input / output) | Used for |
|---|---|---|---|
| Primary extractor | `claude-haiku-4-5` | $1.00 / $5.00 | `appliance_type`, `problem_category`, `part_replaced` |
| Fallback extractor | `claude-haiku-5-5` | $0.10 / $0.50 | The same fields, when the primary fails on a note or can't be used |
| Development assistant | Claude Opus 5.5, via Claude Code | n/a | Pair programming and review while building this; it is not part of the pipeline |

Both runtime models can be changed in `.env` with `LARKSPUR_MODEL` and `LARKSPUR_FALLBACK_MODEL`.

These Claude API features are used:

| Feature | What it does here |
|---|---|
| Messages API | One call per note. |
| Structured outputs (`output_config.format` with a JSON schema) | The model can only return labels allowed by `schema.json`. |
| Token usage on each response | Feeds the measured cost in `outputs/run_stats.json`. |

## Techniques

| Technique | How it is used |
|---|---|
| Hybrid extraction | Deterministic regex and phrase rules handle the fields that follow a pattern (`model_number`, `visit_date`, `follow_up_needed`). The language model handles the three fields that need judgment. |
| Fallback chain | Each note tries the primary model, then the fallback model, then the rule-based answer. |
| Evaluation | Per-field accuracy and confusion matrices on the 50 labeled notes, plus a model-versus-rules agreement check on all 500 notes. |

## Data formats

| Format | Files |
|---|---|
| JSON Lines (`.jsonl`) | Input notes, labels, predictions and diagnostics. |
| JSON | `schema.json` and `outputs/run_stats.json`. |
