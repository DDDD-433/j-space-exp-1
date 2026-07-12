"""Fitting-corpus loading: JSONL/text files or Hugging Face datasets.

Prompt ordering is deterministic given a seed, so fits are reproducible and
shards are disjoint by construction.
"""

from __future__ import annotations

import json
import random
from pathlib import Path


def load_prompts_from_jsonl(path: str | Path, *, text_key: str = "text") -> list[str]:
    """Load prompts from a ``.jsonl`` file (one JSON object per line).

    Each line must contain ``text_key`` (default ``"text"``); lines that are
    plain JSON strings are also accepted.
    """
    prompts: list[str] = []
    for line_no, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        record = json.loads(line)
        if isinstance(record, str):
            prompts.append(record)
        elif isinstance(record, dict) and text_key in record:
            prompts.append(str(record[text_key]))
        else:
            raise ValueError(
                f"{path}:{line_no}: expected a JSON string or an object with a {text_key!r} field"
            )
    return prompts


def load_wikitext_prompts(n_prompts: int, *, min_chars: int = 600) -> list[str]:
    """First ``n_prompts`` WikiText-103 records of >= ``min_chars`` characters,
    streamed from the Hugging Face Hub (requires the ``datasets`` extra)."""
    if n_prompts <= 0:
        return []
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise ImportError(
            "loading HF datasets requires the 'datasets' extra: pip install 'openjspace[datasets]'"
        ) from exc
    dataset = load_dataset(
        "Salesforce/wikitext", "wikitext-103-raw-v1", split="train", streaming=True
    )
    prompts: list[str] = []
    for record in dataset:
        text = record["text"]
        if len(text.strip()) >= min_chars:
            prompts.append(text)
            if len(prompts) == n_prompts:
                break
    return prompts


def load_fitting_prompts(
    dataset: str,
    *,
    num_prompts: int,
    seed: int = 0,
    shard_index: int = 0,
    num_shards: int = 1,
    min_chars: int = 200,
) -> list[str]:
    """Resolve a dataset spec to a deterministic, optionally sharded prompt list.

    Args:
        dataset: Path to a ``.jsonl``/``.txt`` file, or the string
            ``"wikitext"`` for streamed WikiText-103.
        num_prompts: Total prompts across all shards.
        seed: Shuffle seed (deterministic ordering).
        shard_index: This shard's index in ``[0, num_shards)``.
        num_shards: Number of disjoint shards (round-robin split after the
            seeded shuffle, so shards are disjoint and reproducible).
        min_chars: Drop prompts shorter than this many characters.

    Raises:
        ValueError: On bad shard settings or unreadable dataset spec.
    """
    if not 0 <= shard_index < num_shards:
        raise ValueError(f"shard_index {shard_index} out of range for {num_shards} shards")
    path = Path(dataset)
    if dataset == "wikitext":
        prompts = load_wikitext_prompts(num_prompts, min_chars=max(min_chars, 600))
    elif path.suffix == ".jsonl" and path.is_file():
        prompts = load_prompts_from_jsonl(path)
    elif path.suffix in (".txt", "") and path.is_file():
        # Plain text: blank-line-separated documents.
        blocks = Path(path).read_text(encoding="utf-8").split("\n\n")
        prompts = [block.strip() for block in blocks if block.strip()]
    else:
        raise ValueError(f"dataset {dataset!r} not found; pass a .jsonl/.txt path or 'wikitext'")
    prompts = [p for p in prompts if len(p) >= min_chars]
    rng = random.Random(seed)
    rng.shuffle(prompts)
    prompts = prompts[:num_prompts]
    return prompts[shard_index::num_shards]
