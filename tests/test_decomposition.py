"""Tests for sparse non-negative J-space decomposition."""

import itertools

import pytest
import torch

from openjspace.core.decomposition import nonnegative_omp


def test_recovers_known_sparse_combination():
    torch.manual_seed(0)
    d, n = 32, 64
    dictionary = torch.randn(d, n)
    true_support = [3, 17, 40]
    true_coeffs = torch.tensor([2.0, 1.5, 0.7])
    target = sum(c * dictionary[:, i] for c, i in zip(true_coeffs, true_support, strict=True))
    result = nonnegative_omp(dictionary, target, k=5)
    assert set(result.indices) >= set(true_support[:2])  # dominant atoms found
    assert result.reconstruction_error < 1e-3
    assert result.explained_norm_fraction > 0.999
    recovered = dict(zip(result.indices, result.coefficients, strict=True))
    for idx, coeff in zip(true_support, true_coeffs.tolist(), strict=True):
        assert idx in recovered
        assert abs(recovered[idx] - coeff) < 1e-3


def test_enforces_sparsity_budget():
    torch.manual_seed(1)
    dictionary = torch.randn(16, 100)
    target = torch.randn(16)
    for k in (1, 3, 8):
        result = nonnegative_omp(dictionary, target, k=k)
        assert len(result.indices) <= k
        assert result.n_iterations <= k


def test_never_returns_negative_coefficients():
    torch.manual_seed(2)
    for _trial in range(10):
        dictionary = torch.randn(24, 80)
        target = torch.randn(24)
        result = nonnegative_omp(dictionary, target, k=10)
        assert all(c >= 0 for c in result.coefficients)


def test_error_decreases_with_budget():
    torch.manual_seed(3)
    dictionary = torch.randn(32, 128)
    # Target inside the non-negative cone so more atoms genuinely help.
    coeffs = torch.rand(6)
    target = dictionary[:, :6] @ coeffs
    errors = [nonnegative_omp(dictionary, target, k=k).reconstruction_error for k in (1, 2, 4, 6)]
    for previous, current in itertools.pairwise(errors):
        assert current <= previous + 1e-6


def test_correlated_dictionary_does_not_crash():
    torch.manual_seed(4)
    base = torch.randn(16, 8)
    # Highly correlated, overcomplete: many near-duplicate columns.
    dictionary = torch.cat([base + 0.01 * torch.randn(16, 8) for _ in range(12)], dim=1)
    target = base[:, 0] + 0.5 * base[:, 3]
    result = nonnegative_omp(dictionary, target, k=10)
    assert all(c >= 0 for c in result.coefficients)
    assert result.reconstruction_error < float(target.norm())
    assert 0.0 <= result.explained_norm_fraction <= 1.0


def test_zero_target():
    dictionary = torch.randn(8, 20)
    result = nonnegative_omp(dictionary, torch.zeros(8), k=5)
    assert result.indices == []
    assert result.reconstruction_error == 0.0


def test_invalid_arguments():
    dictionary = torch.randn(8, 20)
    with pytest.raises(ValueError, match="k must be positive"):
        nonnegative_omp(dictionary, torch.randn(8), k=0)
    with pytest.raises(ValueError, match="dictionary"):
        nonnegative_omp(torch.randn(8), torch.randn(8), k=2)
    with pytest.raises(ValueError, match="dim"):
        nonnegative_omp(dictionary, torch.randn(9), k=2)


def test_decompose_cell_end_to_end():
    from openjspace.analysis.decompose_runner import decompose_cell
    from openjspace.core.fitting import fit
    from openjspace.models.tiny import TinyAdapter

    adapter = TinyAdapter()
    lens = fit(adapter, ["abcdefghij " * 5], source_layers=[0, 2], dim_batch=4, max_seq_len=64)
    record = decompose_cell(adapter, lens, "the quick brown fox", layer=2, position=-1, k=4)
    assert record.layer == 2
    assert len(record.entries) <= 4
    assert all(e.coefficient >= 0 for e in record.entries)
    assert 0.0 <= record.explained_norm_fraction <= 1.0
    assert "non-unique" in record.warning
    with pytest.raises(ValueError, match="not fitted"):
        decompose_cell(adapter, lens, "the quick brown fox", layer=1, position=-1, k=4)
    with pytest.raises(ValueError, match="out of range"):
        decompose_cell(adapter, lens, "the quick brown fox", layer=2, position=999, k=4)
