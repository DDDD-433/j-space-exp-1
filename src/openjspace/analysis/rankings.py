# Portions derived from anthropics/jacobian-lens (jlens/vis.py, _ranks_of)
# Copyright 2026 Anthropic PBC
# Copyright 2026 The OpenJSpace Authors
# SPDX-License-Identifier: Apache-2.0
"""Rank computation and token display helpers."""

from __future__ import annotations

import unicodedata

import torch


def ranks_of(
    logits: torch.Tensor, target_ids: torch.Tensor, *, chunk_size: int = 256
) -> torch.Tensor:
    """Full-vocab ranks of ``target_ids`` at every position, chunked over the
    sequence so peak memory is one ``[chunk_size, vocab]`` sort buffer.

    Args:
        logits: ``[seq_len, vocab]``.
        target_ids: 1-D ``[n_targets]`` (same targets at every position) or
            2-D ``[seq_len, n_targets]`` (per-position targets).

    Returns:
        ``[seq_len, n_targets]`` int64 ranks (0 = top-ranked).
    """
    seq_len, vocab = logits.shape
    out = torch.empty(seq_len, target_ids.shape[-1], dtype=torch.long, device=logits.device)
    arange = torch.arange(vocab, device=logits.device)
    for start in range(0, seq_len, chunk_size):
        sl = slice(start, start + chunk_size)
        sorted_idx = logits[sl].argsort(dim=-1, descending=True)
        full_rank = torch.empty_like(sorted_idx)
        full_rank.scatter_(1, sorted_idx, arange.expand_as(sorted_idx))
        idx = target_ids if target_ids.ndim == 1 else target_ids[sl]
        out[sl] = full_rank.gather(1, idx.expand(full_rank.shape[0], -1))
        del sorted_idx, full_rank
    return out


def normalized_scores(logits: torch.Tensor, top_values: torch.Tensor) -> torch.Tensor:
    """Map top-K logit values to a [0, 1] display scale per row.

    This is a *display normalization* (min-max against the row's logit spread),
    not a probability: lens logits are uncalibrated and the UI should show
    rankings or normalized scores rather than implying probabilities.
    """
    row_max = logits.max(dim=-1, keepdim=True).values
    row_min = logits.min(dim=-1, keepdim=True).values
    span = (row_max - row_min).clamp_min(1e-9)
    return ((top_values - row_min) / span).clamp(0.0, 1.0)


def clean_token_text(raw: str) -> str:
    """Human-friendly display form of a raw tokenizer string.

    Replaces leading space markers and control characters with visible
    equivalents. The raw string must always remain available elsewhere; this
    is for display only.
    """
    text = raw.replace("\u0120", " ").replace("\u2581", " ")
    cleaned = []
    for ch in text:
        if ch == "\n":
            cleaned.append("\\n")
        elif ch == "\t":
            cleaned.append("\\t")
        elif unicodedata.category(ch).startswith("C"):
            cleaned.append("\ufffd")
        else:
            cleaned.append(ch)
    return "".join(cleaned)


def is_wordlike(raw: str) -> bool:
    """True when a decoded token is made of word characters (used by display
    filters; never applied to stored data)."""
    stripped = raw.strip()
    if not stripped:
        return False
    if "<|" in stripped or (stripped.startswith("<") and stripped.endswith(">")):
        return False
    return all(
        ch.isalnum() or (0 < pos < len(stripped) - 1 and ch in "'-\u2019")
        for pos, ch in enumerate(stripped)
    )
