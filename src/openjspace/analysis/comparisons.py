"""Comparing readout methods: J-lens vs. logit lens vs. actual final logits.

None of these methods is universally "correct": the J-lens corrects for
representational drift across layers but is a corpus-averaged linear
approximation; the logit lens is exact only at the final layer; the model's
final logits describe the output distribution, not intermediate content.
"""

from __future__ import annotations

import torch


def topk_overlap(ids_a: torch.Tensor, ids_b: torch.Tensor) -> float:
    """Jaccard-free overlap fraction: ``|A ∩ B| / k`` for two top-K id lists."""
    set_a = set(ids_a.flatten().tolist())
    set_b = set(ids_b.flatten().tolist())
    k = max(len(set_a), 1)
    return len(set_a & set_b) / k


def spearman_rank_correlation(scores_a: torch.Tensor, scores_b: torch.Tensor) -> float:
    """Spearman rank correlation between two score vectors over the same ids.

    Args:
        scores_a: ``[n]`` scores from method A.
        scores_b: ``[n]`` scores from method B (same item order).

    Returns:
        Correlation in [-1, 1]; 0.0 when fewer than two items.
    """
    n = scores_a.numel()
    if n < 2:
        return 0.0

    def to_ranks(scores: torch.Tensor) -> torch.Tensor:
        order = scores.argsort(descending=True)
        ranks = torch.empty_like(order, dtype=torch.float64)
        ranks[order] = torch.arange(n, dtype=torch.float64)
        return ranks

    ranks_a = to_ranks(scores_a.double().flatten())
    ranks_b = to_ranks(scores_b.double().flatten())
    ranks_a = ranks_a - ranks_a.mean()
    ranks_b = ranks_b - ranks_b.mean()
    denom = ranks_a.norm() * ranks_b.norm()
    if denom == 0:
        return 0.0
    return float((ranks_a @ ranks_b) / denom)


def compare_cell(
    jlens_logits: torch.Tensor,
    logit_lens_logits: torch.Tensor,
    *,
    top_k: int,
) -> dict[str, float]:
    """Per-cell comparison metrics between the two lens readouts.

    Args:
        jlens_logits: ``[vocab]`` J-lens logits at one (layer, position).
        logit_lens_logits: ``[vocab]`` logit-lens logits at the same cell.
        top_k: Size of the top set used for overlap.

    Returns:
        ``{"topk_overlap": ..., "rank_correlation": ...}`` where the rank
        correlation is computed over the union of the two top-K sets.
    """
    top_j = jlens_logits.topk(top_k).indices
    top_l = logit_lens_logits.topk(top_k).indices
    overlap = topk_overlap(top_j, top_l)
    union = torch.unique(torch.cat([top_j, top_l]))
    corr = spearman_rank_correlation(jlens_logits[union], logit_lens_logits[union])
    return {"topk_overlap": overlap, "rank_correlation": corr}
