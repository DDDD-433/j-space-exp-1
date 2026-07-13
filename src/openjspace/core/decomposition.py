"""Sparse non-negative decomposition of an activation in J-lens directions.

Implements non-negative orthogonal matching pursuit (NN-OMP), a greedy
non-negative pursuit in the same family as the gradient pursuit used in the
paper: at each step the dictionary atom with the largest positive correlation
to the residual is added, then all active coefficients are refit by
non-negative least squares on the active set.

Given an activation ``h`` and dictionary ``D`` (J-lens vectors as columns),
approximately solve::

    min ||h - D_S a||_2   s.t.  a >= 0,  |S| <= K

The dictionary is correlated and overcomplete, so the decomposition is
NON-UNIQUE: results are one sparse representative, not a canonical
decomposition. Top-k-by-inner-product is *not* a decomposition and is never
labeled as one.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class SparseDecomposition:
    """Result of one non-negative pursuit run.

    Attributes:
        indices: Selected dictionary column indices, in selection order.
        coefficients: Non-negative coefficients aligned with ``indices``.
        reconstruction: ``D_S a`` in activation space.
        reconstruction_error: ``||h - D_S a||_2``.
        residual_norm: Alias of ``reconstruction_error`` (final residual).
        explained_norm_fraction: ``1 - (residual / ||h||)^2`` clamped to [0,1];
            the fraction of squared activation norm captured.
        n_iterations: Number of greedy steps actually taken.
    """

    indices: list[int]
    coefficients: list[float]
    reconstruction: torch.Tensor
    reconstruction_error: float
    residual_norm: float
    explained_norm_fraction: float
    n_iterations: int


def _nnls_on_support(D_S: torch.Tensor, h: torch.Tensor, *, inner_iters: int = 200) -> torch.Tensor:
    """Non-negative least squares ``min ||h - D_S a||, a >= 0`` on a small
    active set, via projected gradient with Lipschitz step size.

    Args:
        D_S: ``[d, s]`` active dictionary columns (s is small, <= K).
        h: ``[d]`` target.

    Returns:
        ``[s]`` non-negative coefficients.
    """
    gram = D_S.T @ D_S  # [s, s]
    correlation = D_S.T @ h  # [s]
    # Lipschitz constant of the gradient = largest eigenvalue of the Gram.
    lipschitz = torch.linalg.eigvalsh(gram)[-1].clamp_min(1e-12)
    step = 1.0 / float(lipschitz)
    a = torch.clamp(torch.linalg.lstsq(gram, correlation).solution, min=0.0)
    for _ in range(inner_iters):
        grad = gram @ a - correlation
        a_next = torch.clamp(a - step * grad, min=0.0)
        if torch.max(torch.abs(a_next - a)) < 1e-8:
            a = a_next
            break
        a = a_next
    return a


def nonnegative_omp(
    dictionary: torch.Tensor,
    target: torch.Tensor,
    *,
    k: int,
    tol: float = 1e-6,
) -> SparseDecomposition:
    """Greedy non-negative orthogonal matching pursuit.

    Args:
        dictionary: ``[d, n_atoms]`` dictionary; columns need not be
            normalized (selection uses correlation with unit-normalized
            atoms so scale differences don't bias the greedy choice).
        target: ``[d]`` activation to decompose.
        k: Maximum number of atoms (sparsity budget).
        tol: Stop early when the residual norm falls below ``tol`` or no atom
            has positive correlation with the residual.

    Returns:
        A :class:`SparseDecomposition` with at most ``k`` atoms and strictly
        non-negative coefficients.

    Raises:
        ValueError: On empty dictionary or non-positive ``k``.
    """
    if dictionary.ndim != 2 or dictionary.shape[1] == 0:
        raise ValueError(f"dictionary must be [d, n_atoms], got {tuple(dictionary.shape)}")
    if k <= 0:
        raise ValueError(f"k must be positive, got {k}")
    target = target.float().flatten()
    if target.shape[0] != dictionary.shape[0]:
        raise ValueError(f"target dim {target.shape[0]} != dictionary dim {dictionary.shape[0]}")

    atom_norms = dictionary.norm(dim=0).clamp_min(1e-12)
    target_norm = float(target.norm())
    residual = target.clone()
    support: list[int] = []
    coefficients = torch.zeros(0)

    for _ in range(k):
        if float(residual.norm()) <= tol:
            break
        correlation = (dictionary.T @ residual) / atom_norms
        if support:
            correlation[torch.tensor(support, dtype=torch.long)] = float("-inf")
        best = int(correlation.argmax())
        if float(correlation[best]) <= 0:
            break  # no atom can reduce the residual with a non-negative coefficient
        support.append(best)
        D_S = dictionary[:, torch.tensor(support, dtype=torch.long)]
        coefficients = _nnls_on_support(D_S, target)
        residual = target - D_S @ coefficients

    if support:
        # Drop atoms whose refit coefficient collapsed to zero.
        keep = [i for i, c in enumerate(coefficients.tolist()) if c > 0]
        support = [support[i] for i in keep]
        coefficients = (
            coefficients[torch.tensor(keep, dtype=torch.long)] if keep else torch.zeros(0)
        )
        if support:
            D_S = dictionary[:, torch.tensor(support, dtype=torch.long)]
            reconstruction = D_S @ coefficients
            residual = target - reconstruction
        else:
            reconstruction = torch.zeros_like(target)
            residual = target.clone()
    else:
        reconstruction = torch.zeros_like(target)

    error = float(residual.norm())
    explained = 0.0
    if target_norm > 0:
        explained = max(0.0, min(1.0, 1.0 - (error / target_norm) ** 2))
    return SparseDecomposition(
        indices=support,
        coefficients=[float(c) for c in coefficients.tolist()],
        reconstruction=reconstruction,
        reconstruction_error=error,
        residual_norm=error,
        explained_norm_fraction=explained,
        n_iterations=len(support),
    )
