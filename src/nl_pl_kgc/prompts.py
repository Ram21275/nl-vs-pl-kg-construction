from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Literal, Sequence

from .data import Example, Triple

PromptFormat = Literal["natural", "code"]

CODE_SCHEMA = '''from dataclasses import dataclass

@dataclass(frozen=True)
class Triple:
    head: str
    relation: str
    tail: str

def extract(text: str) -> list[Triple]:
    """Return every knowledge-graph triple explicitly supported by text."""'''


@dataclass(frozen=True)
class PromptPair:
    system: str
    user: str


def format_triples(triples: Sequence[Triple], prompt_format: PromptFormat) -> str:
    if prompt_format == "natural":
        return json.dumps([list(t) for t in triples], ensure_ascii=False)
    if prompt_format == "code":
        if not triples:
            return "return []"
        rendered = ",\n        ".join(
            f"Triple({head!r}, {relation!r}, {tail!r})" for head, relation, tail in triples
        )
        return f"return [\n        {rendered}\n    ]"
    raise ValueError(f"Unknown prompt format: {prompt_format}")


def build_prompt(
    text: str,
    prompt_format: PromptFormat,
    demonstrations: Sequence[Example] = (),
) -> PromptPair:
    """Create matched prompts whose semantic task is constant across formats."""
    if prompt_format == "natural":
        system = (
            "You extract knowledge graph triples from text. Return only a JSON array. "
            "Each item must be [head, relation, tail]. Do not infer unsupported facts."
        )
        blocks = []
        for item in demonstrations:
            blocks.append(
                f"Example text:\n{item.text}\nExample triples:\n"
                f"{format_triples(item.triples, 'natural')}"
            )
        blocks.append(f"Text:\n{text}\nTriples:")
        return PromptPair(system=system, user="\n\n".join(blocks))

    if prompt_format == "code":
        system = (
            "You are a Python programming assistant that extracts knowledge graph triples. "
            "Complete only the function body with a return statement; do not add explanation."
        )
        blocks = [CODE_SCHEMA]
        for item in demonstrations:
            blocks.append(
                f"\n    # Example input: {item.text!r}\n    "
                + format_triples(item.triples, "code").replace("\n", "\n    ")
            )
        blocks.append(f"\n    # Input: {text!r}\n    ")
        return PromptPair(system=system, user="\n".join(blocks))

    raise ValueError(f"Unknown prompt format: {prompt_format}")


def training_text(
    tokenizer,
    example: Example,
    prompt_format: PromptFormat,
    demonstrations: Sequence[Example],
) -> str:
    pair = build_prompt(example.text, prompt_format, demonstrations)
    answer = format_triples(example.triples, prompt_format)
    messages = [
        {"role": "system", "content": pair.system},
        {"role": "user", "content": pair.user},
        {"role": "assistant", "content": answer},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
