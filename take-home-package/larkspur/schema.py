"""Field names and allowed values, read from data/schema.json.

schema.json is the authority on the label sets, so the enums are loaded from
it rather than copied into the code.
"""
from __future__ import annotations

import json

from larkspur.config import SCHEMA_PATH

with open(SCHEMA_PATH, encoding="utf-8") as _f:
    _SCHEMA = json.load(_f)["extract_fields"]

# Output column order follows schema.json.
FIELDS: list[str] = list(_SCHEMA)

APPLIANCE_TYPES: list[str] = _SCHEMA["appliance_type"]["allowed"]
PROBLEM_CATEGORIES: list[str] = _SCHEMA["problem_category"]["allowed"]
PARTS: list[str] = _SCHEMA["part_replaced"]["allowed"]  # includes "none"

RULE_FIELDS = ["model_number", "visit_date", "follow_up_needed"]
LLM_FIELDS = ["appliance_type", "problem_category", "part_replaced"]

assert set(RULE_FIELDS) | set(LLM_FIELDS) == set(FIELDS), "schema.json fields changed"
