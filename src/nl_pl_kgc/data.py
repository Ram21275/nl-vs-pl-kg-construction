from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

Triple = tuple[str, str, str]


@dataclass(frozen=True)
class Example:
    text: str
    triples: tuple[Triple, ...]


def _normalise_triple(value: Iterable[str]) -> Triple:
    items = tuple(str(part) for part in value)
    if len(items) != 3:
        raise ValueError(f"Expected a three-part triple, received {items!r}")
    return items  # type: ignore[return-value]


def load_examples(path: str | Path) -> list[Example]:
    """Load the paper repository's JSON format without pandas."""
    source = Path(path)
    records = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(records, list):
        raise ValueError(f"Expected a JSON list in {source}")

    examples: list[Example] = []
    for index, record in enumerate(records):
        if not isinstance(record, dict) or "text" not in record:
            raise ValueError(f"Invalid record {index} in {source}")
        raw_triples = record.get("triple_list", record.get("triples", []))
        examples.append(
            Example(
                text=str(record["text"]),
                triples=tuple(_normalise_triple(item) for item in raw_triples),
            )
        )
    return examples


def save_examples(path: str | Path, examples: Iterable[Example]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = [
        {"text": item.text, "triple_list": [list(triple) for triple in item.triples]}
        for item in examples
    ]
    target.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
