"""Loading inspection prompts (single strings or example files)."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ExamplePrompt:
    """One qualitative-evaluation example.

    Attributes:
        slug: Identifier.
        description: What the example is probing.
        prompt: Raw text prompt (used when ``user`` is not set).
        user: Chat-mode user message (formatted via the chat template).
        expected_concepts: Qualitative concepts a reader might look for; these
            are hypotheses to check, not guaranteed readouts.
    """

    slug: str
    description: str
    prompt: str | None = None
    user: str | None = None
    expected_concepts: tuple[str, ...] = ()


def load_examples_jsonl(path: str | Path) -> list[ExamplePrompt]:
    """Load examples from a JSONL file with keys matching :class:`ExamplePrompt`."""
    examples: list[ExamplePrompt] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        record = json.loads(line)
        examples.append(
            ExamplePrompt(
                slug=record["slug"],
                description=record.get("description", ""),
                prompt=record.get("prompt"),
                user=record.get("user"),
                expected_concepts=tuple(record.get("expected_concepts", [])),
            )
        )
    return examples
