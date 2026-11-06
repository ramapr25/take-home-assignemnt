"""Run the extraction pipeline.

    python run.py              # rules + LLM -> predictions.jsonl, then evaluate
    python run.py --dry-run    # rules only, no API calls
    python run.py --help       # all options
"""
import sys

from larkspur.pipeline import main

if __name__ == "__main__":
    sys.exit(main())
