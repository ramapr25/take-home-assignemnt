# How to run

This takes about five minutes on a fresh machine. A full run calls the Claude API and costs about $0.70. The dry run in step 4 is free and needs no API key.

## 1. Prerequisites

- **Python 3.10 or newer.** Check with `python --version` (on macOS or Linux, `python3 --version`).
- **An Anthropic API key**, from https://console.anthropic.com. It's only needed for the full run.

## 2. Set up

Open a terminal in this folder, the one that contains `run.py`, and run the commands for your system.

**macOS / Linux**
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

**Windows (PowerShell)**
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

If PowerShell refuses to run the activate script, run this once and then try the activate command again:
```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

## 3. Create a `.env` file

Create a file named `.env` in this folder, next to `run.py`, with these lines:

```
# Required for the full run (not needed for --dry-run).
ANTHROPIC_API_KEY=sk-ant-...

# Primary model, used for every note.
LARKSPUR_MODEL=claude-haiku-4-5
# Fallback model, tried when the primary fails. "none" skips straight to the rule answers.
LARKSPUR_FALLBACK_MODEL=claude-haiku-5-5
```

Replace `sk-ant-...` with your key. To switch models later, edit the two `LARKSPUR_` lines; no code changes are needed. Both model lines are optional, and the values above are the defaults.

Instead of the file, you can set these as environment variables in your shell. If both are set, the shell value wins. A `--model` or `--fallback-model` flag on the command line overrides both for that run.

Never share or commit `.env`, because it holds your key.

## 4. Run

| Command | What it does | Needs a key? |
|---|---|---|
| `python run.py` | Full pipeline: rules plus Claude Haiku 4.5 on all 500 notes. Writes `predictions.jsonl` and prints the accuracy report. | Yes, about $0.70 |
| `python run.py --dry-run` | Rules only, no API calls. Writes `outputs/predictions_dryrun.jsonl` and prints the accuracy report. | No |
| `python run.py --only-labeled` | Runs Claude on just the 50 labeled notes, which is useful for quick checks. Writes `outputs/predictions_partial.jsonl`. | Yes, about $0.07 |

Useful options:
- `--model claude-haiku-5-5` switches the main model. You can also set `LARKSPUR_MODEL` in `.env`.
- `--fallback-model <id>` sets the model tried when the main one fails. The default is `claude-haiku-5-5`. Use `none` to go straight to the rule answers. You can also set `LARKSPUR_FALLBACK_MODEL` in `.env`.
- `--workers N` sets how many API calls run in parallel. The default is 8.
- `--limit N` processes only the first N notes.
- `--no-eval` skips the accuracy report.

## 5. What you should see

A full run takes about one to two minutes. It logs progress every 50 notes, prints a JSON block with the token usage and cost, and then prints:

- accuracy per field on the 50 labeled notes;
- confusion matrices for the three categorical fields;
- how often Claude and the rule baseline agree across all 500 notes.

Output files:

| File | Contents |
|---|---|
| `predictions.jsonl` | The deliverable: one line per note, with `note_id` and the six fields. |
| `outputs/run_stats.json` | Token counts and cost for the last full run. |
| `outputs/errors.jsonl` | Every field that disagrees with the labels, with the note text. |
| `outputs/disagreements.jsonl` | Every note where Claude and the rules disagree. |
| `outputs/predictions_rules.jsonl` | The rule-only answers for all six fields. |

## Troubleshooting

- **`ANTHROPIC_API_KEY is not set`.** Check that `.env` is in the same folder as `run.py` and that the line has no quotes or spaces around `=`. If you don't have a key, use `--dry-run`.
- **`AuthenticationError` or `PermissionDeniedError`.** The key is wrong, or its account has no credit.
- **Rate-limit warnings.** The SDK retries these on its own. If they keep happening, lower `--workers`.
- **`ModuleNotFoundError`.** The virtual environment isn't active. Activate it as in step 2 and run `pip install -r requirements.txt` again.
- **The main model fails, or your key can't use it.** The run carries on with the fallback model, and the log says so once. If the fallback fails too, that note gets the rule answer and is listed under `failures` in `outputs/run_stats.json`. `answered_by` in the same file shows how many notes each model or the rules answered.
- **`The API key was rejected`.** The run stops straight away instead of falling back, because every model would fail with the same key.
