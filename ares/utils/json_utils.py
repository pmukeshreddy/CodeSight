from __future__ import annotations

import json
from json import JSONDecodeError


def parse_llm_json(raw: str, fallback: dict | None = None) -> dict:
    """Parse JSON from LLM output, tolerating preamble/postamble text."""
    if fallback is None:
        fallback = {}
    if not raw.strip():
        return dict(fallback)
    try:
        return json.loads(raw)
    except JSONDecodeError:
        start = raw.find("{")
        end = raw.rfind("}")
        if start == -1 or end == -1 or end <= start:
            return dict(fallback)
        try:
            return json.loads(raw[start : end + 1])
        except JSONDecodeError:
            return dict(fallback)
