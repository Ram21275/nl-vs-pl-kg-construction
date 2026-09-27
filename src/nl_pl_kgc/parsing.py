from __future__ import annotations

import ast
import json
import re
from typing import Iterable

from .data import Triple

_TRIPLE_CALL = re.compile(
    r"Triple\(\s*(['\"])(.*?)\1\s*,\s*(['\"])(.*?)\3\s*,\s*(['\"])(.*?)\5\s*\)",
    flags=re.DOTALL,
)


def _as_triples(value: object) -> list[Triple]:
    if not isinstance(value, (list, tuple)):
        return []
    triples: list[Triple] = []
    for item in value:
        if isinstance(item, (list, tuple)) and len(item) == 3:
            triples.append(tuple(str(part).strip() for part in item))  # type: ignore[arg-type]
    return triples


def parse_natural_output(text: str) -> list[Triple]:
    """Parse a strict JSON result, with a safe literal fallback for minor quote drift."""
    candidates = [text.strip()]
    start, end = text.find("["), text.rfind("]")
    if start >= 0 and end > start:
        candidates.append(text[start : end + 1])
    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            try:
                parsed = ast.literal_eval(candidate)
            except (ValueError, SyntaxError):
                continue
        triples = _as_triples(parsed)
        if triples or parsed == []:
            return triples
    return []


def parse_code_output(text: str) -> list[Triple]:
    """Parse only literal Triple(...) calls; model-produced code is never executed."""
    return [(match[2].strip(), match[4].strip(), match[6].strip()) for match in _TRIPLE_CALL.finditer(text)]


def parse_output(text: str, prompt_format: str) -> list[Triple]:
    if prompt_format == "natural":
        return parse_natural_output(text)
    if prompt_format == "code":
        return parse_code_output(text)
    raise ValueError(f"Unknown prompt format: {prompt_format}")


def deduplicate(triples: Iterable[Triple]) -> list[Triple]:
    return list(dict.fromkeys(triples))
