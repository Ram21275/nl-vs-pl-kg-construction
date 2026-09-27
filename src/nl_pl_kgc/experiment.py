from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Sequence

from .data import Example, load_examples
from .metrics import strict_micro_metrics
from .parsing import parse_output
from .prompts import PromptFormat, build_prompt


def choose_demonstrations(
    train: Sequence[Example], count: int, seed: int, exclude: Example | None = None
) -> list[Example]:
    candidates = [item for item in train if item != exclude]
    if count > len(candidates):
        raise ValueError(f"Requested {count} demonstrations from {len(candidates)} examples")
    return random.Random(seed).sample(candidates, count)


def _mock_response(example: Example, prompt_format: PromptFormat) -> str:
    """Deterministic smoke-test backend; it does not represent model performance."""
    if prompt_format == "natural":
        return json.dumps([list(t) for t in example.triples])
    calls = ", ".join(f"Triple({h!r}, {r!r}, {t!r})" for h, r, t in example.triples)
    return f"return [{calls}]"


def run_mock_evaluation(dataset_path: str | Path, output_dir: str | Path) -> dict:
    examples = load_examples(dataset_path)
    results: dict[str, dict] = {}
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    for prompt_format in ("natural", "code"):
        predictions = []
        records = []
        for item in examples:
            prompt = build_prompt(item.text, prompt_format)
            raw = _mock_response(item, prompt_format)
            parsed = parse_output(raw, prompt_format)
            predictions.append(parsed)
            records.append(
                {
                    "text": item.text,
                    "system": prompt.system,
                    "prompt": prompt.user,
                    "raw_output": raw,
                    "prediction": [list(t) for t in parsed],
                    "gold": [list(t) for t in item.triples],
                }
            )
        metrics = strict_micro_metrics([e.triples for e in examples], predictions)
        results[prompt_format] = metrics.as_dict()
        (out / f"mock_{prompt_format}_predictions.json").write_text(
            json.dumps(records, indent=2, ensure_ascii=False), encoding="utf-8"
        )
    (out / "mock_metrics.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    return results
