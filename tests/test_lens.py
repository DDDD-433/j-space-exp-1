"""Tests for lens application and the inspection pipeline."""

import pytest
import torch

from openjspace.core.applying import inspect_prompt, select_positions
from openjspace.core.fitting import fit
from openjspace.models.tiny import TinyAdapter

PROMPTS = ["abcdefghij " * 5, "klmnopqrst " * 5]


@pytest.fixture(scope="module")
def fitted():
    adapter = TinyAdapter()
    lens = fit(adapter, PROMPTS, source_layers=[0, 1, 2], dim_batch=4, max_seq_len=64)
    return adapter, lens


def test_select_positions_specs():
    assert select_positions("all", 5) == [0, 1, 2, 3, 4]
    assert select_positions("last:2", 5) == [3, 4]
    assert select_positions("0,-1", 5) == [0, 4]
    assert select_positions([1, -2], 5) == [1, 3]
    with pytest.raises(ValueError, match="out of range"):
        select_positions([99], 5)
    with pytest.raises(ValueError, match="not understood"):
        select_positions("frogs", 5)


def test_inspect_produces_grid(fitted):
    adapter, lens = fitted
    result = inspect_prompt(adapter, lens, "the quick brown fox", top_k=5)
    seq_len = len(result.positions)
    # 3 lens layers + final model-output row.
    assert result.metadata.layers == [0, 1, 2, 3]
    assert len(result.cells) == 4 * seq_len
    for cell in result.cells:
        assert len(cell.jlens_top) == 5
        assert len(cell.logit_lens_top) == 5
        assert cell.activation_norm > 0
        for entry in cell.jlens_top:
            assert 0.0 <= entry.normalized_score <= 1.0
    output_rows = [c for c in result.cells if c.is_model_output]
    assert all(c.layer == 3 for c in output_rows)


def test_final_row_jlens_equals_logit_lens(fitted):
    """On the model-output row (J = I) the two readouts coincide."""
    adapter, lens = fitted
    result = inspect_prompt(adapter, lens, "the quick brown fox", top_k=5)
    for cell in result.cells:
        if cell.is_model_output:
            assert [e.token_id for e in cell.jlens_top] == [e.token_id for e in cell.logit_lens_top]


def test_tiny_linear_model_jlens_matches_output(fitted):
    """The tiny model's blocks are exactly linear, so the transported layer-2
    readout must equal the model output readout at every position."""
    adapter, lens = fitted
    result = inspect_prompt(adapter, lens, "the quick brown fox jumps", top_k=3)
    by_pos_l2 = {c.position: c for c in result.cells if c.layer == 2}
    by_pos_out = {c.position: c for c in result.cells if c.is_model_output}
    for pos, cell in by_pos_l2.items():
        assert [e.token_id for e in cell.jlens_top] == [
            e.token_id for e in by_pos_out[pos].jlens_top
        ]


def test_positions_and_layers_subsets(fitted):
    adapter, lens = fitted
    result = inspect_prompt(
        adapter, lens, "the quick brown fox", layers=[0, 2], positions="last:3", top_k=4
    )
    assert result.metadata.layers == [0, 2, 3]
    assert len({c.position for c in result.cells}) == 3


def test_unfitted_layer_rejected(fitted):
    adapter, lens = fitted
    with pytest.raises(ValueError, match="not fitted"):
        inspect_prompt(adapter, lens, "the quick brown fox", layers=[3])


def test_tracked_concepts_ranks_and_scores(fitted):
    adapter, lens = fitted
    result = inspect_prompt(adapter, lens, "the quick brown fox", top_k=3, tracked_token_ids=[3, 7])
    assert len(result.tracked) == 2
    n_rows = len(result.metadata.layers)
    n_positions = result.metadata.n_positions
    for tracked in result.tracked:
        assert len(tracked.ranks) == n_rows
        assert all(len(row) == n_positions for row in tracked.ranks)
        assert all(0 <= r < adapter.vocab_size for row in tracked.ranks for r in row)


def test_comparison_metrics_bounds(fitted):
    adapter, lens = fitted
    result = inspect_prompt(adapter, lens, "the quick brown fox", top_k=5)
    for cell in result.cells:
        assert 0.0 <= cell.comparison["topk_overlap"] <= 1.0
        assert -1.0 <= cell.comparison["rank_correlation"] <= 1.0


def test_transport_shape_and_device(fitted):
    adapter, lens = fitted
    h = torch.randn(7, adapter.hidden_size)
    out = lens.transport(h, 0)
    assert out.shape == (7, adapter.hidden_size)
    with pytest.raises(ValueError, match="not fitted"):
        lens.transport(h, 3)
