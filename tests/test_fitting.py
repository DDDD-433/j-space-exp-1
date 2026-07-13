# Portions derived from anthropics/jacobian-lens (tests/test_fitting.py)
# Copyright 2026 Anthropic PBC
# Copyright 2026 The OpenJSpace Authors
# SPDX-License-Identifier: Apache-2.0
"""Mathematical tests for the fitting estimator."""

import pytest
import torch

from openjspace.core.fitting import (
    FittingCancelled,
    fit,
    jacobian_for_prompt,
    valid_position_mask,
)
from openjspace.core.lens import JacobianLens
from openjspace.models.tiny import TinyAdapter

LONG_A = "abcdefghij " * 5
LONG_B = "klmnopqrst " * 5
LONG_C = "uvwxyzabcd " * 5


def test_valid_position_mask_basic():
    mask = valid_position_mask(32, skip_first=4)
    assert mask.dtype == torch.bool
    assert mask[:4].sum() == 0  # leading attention-sink positions excluded
    assert not mask[-1]  # final position excluded (no next-token target)
    assert mask[4:-1].all()
    assert mask.sum() == 32 - 4 - 1


def test_valid_position_mask_default_skips_16():
    mask = valid_position_mask(64)
    assert mask[:16].sum() == 0
    assert mask.sum() == 64 - 16 - 1


def test_valid_position_mask_too_short():
    with pytest.raises(ValueError, match="too short"):
        valid_position_mask(5, skip_first=8)
    with pytest.raises(ValueError, match="skip_first"):
        valid_position_mask(5, skip_first=-1)


def test_jacobian_for_prompt_tiny_shapes_and_orientation():
    """End-to-end on a 4-layer CPU model: shapes + late-layer diag ~= 1.

    All parameters have requires_grad=False, so the recorder's graph rooting
    must carry the autograd graph itself.
    """
    adapter = TinyAdapter(n_layers=4, hidden=8)
    prompt = "the quick brown fox " * 4
    jacobians, seq_len, n_valid = jacobian_for_prompt(
        adapter, prompt, source_layers=[0, 1, 2], dim_batch=4, max_seq_len=64
    )
    assert set(jacobians) == {0, 1, 2}
    for J in jacobians.values():
        assert J.shape == (8, 8) and J.dtype == torch.float32
        assert torch.isfinite(J).all()
    assert n_valid > 0 and seq_len > n_valid
    # Blocks are h + 0.1*W*h, so J_2 = I + W_3 exactly — pins orientation/indexing.
    expected_J2 = torch.eye(8) + adapter.model.layers[3].linear.weight.detach()
    torch.testing.assert_close(jacobians[2], expected_J2, rtol=0, atol=1e-5)
    # Earlier layers compound through more blocks -> further from identity.
    assert (jacobians[0] - torch.eye(8)).norm() > (jacobians[2] - torch.eye(8)).norm()


def test_jacobian_matches_finite_differences():
    """Autograd Jacobian == central finite differences on the tiny linear net.

    The tiny model has no attention, so ``dh_L[t']/dh_l[t] = 0`` for
    ``t' != t`` and the estimator reduces to the per-position Jacobian of the
    downstream block composition, which finite differences recover exactly.
    This is the only place finite differences are used in the project.
    """
    adapter = TinyAdapter(n_layers=3, hidden=6)
    source_layer = 0
    jacobians, _, _ = jacobian_for_prompt(
        adapter,
        "the quick brown fox jumps over the lazy dog",
        source_layers=[source_layer],
        dim_batch=3,
        max_seq_len=64,
        skip_first=4,
    )
    J_auto = jacobians[source_layer]

    def downstream(h: torch.Tensor) -> torch.Tensor:
        for block in adapter.model.layers[source_layer + 1 :]:
            h = block(h)
        return h

    eps = 1e-3
    h0 = torch.randn(6)
    J_fd = torch.zeros(6, 6)
    with torch.no_grad():
        for i in range(6):
            e = torch.zeros(6)
            e[i] = eps
            J_fd[:, i] = (downstream(h0 + e) - downstream(h0 - e)) / (2 * eps)
    torch.testing.assert_close(J_auto, J_fd, rtol=1e-4, atol=1e-4)


def test_future_position_masking_sums_over_targets():
    """The cotangent covers all valid target positions: with a causal-free
    (position-independent) model, per-source gradients equal the single-
    position Jacobian, which the previous test pins. Here we verify the
    skip-prefix knob changes which positions contribute."""
    adapter = TinyAdapter(n_layers=3, hidden=6)
    prompt = "abcdefghijabcdefghijabcdefghij"
    j_skip4, _, n_valid4 = jacobian_for_prompt(
        adapter, prompt, source_layers=[0], dim_batch=3, max_seq_len=64, skip_first=4
    )
    j_skip8, _, n_valid8 = jacobian_for_prompt(
        adapter, prompt, source_layers=[0], dim_batch=3, max_seq_len=64, skip_first=8
    )
    assert n_valid8 < n_valid4
    # Tiny model is position-independent, so the mean is unchanged.
    torch.testing.assert_close(j_skip4[0], j_skip8[0], rtol=1e-5, atol=1e-6)


def test_fit_deterministic_given_same_prompts():
    prompts = [LONG_A, LONG_B]
    lens1 = fit(TinyAdapter(seed=7), prompts, source_layers=[0, 2], dim_batch=4, max_seq_len=64)
    lens2 = fit(TinyAdapter(seed=7), prompts, source_layers=[0, 2], dim_batch=4, max_seq_len=64)
    for layer in (0, 2):
        torch.testing.assert_close(lens1.jacobians[layer], lens2.jacobians[layer])


def test_fit_metadata_counts():
    lens = fit(
        TinyAdapter(),
        [LONG_A, LONG_B],
        source_layers=[0, 1],
        dim_batch=4,
        max_seq_len=64,
        skip_first=4,
    )
    md = lens.metadata
    assert md.number_of_prompts == 2
    assert md.number_of_valid_positions > 0
    assert md.source_layers == [0, 1]
    assert md.target_layer == 3
    assert md.residual_location == "block_output"
    assert md.dtype_used_for_accumulation == "float32"


def test_fit_checkpoint_resume(tmp_path):
    adapter = TinyAdapter()
    prompts = [LONG_A, LONG_B, LONG_C]
    checkpoint = tmp_path / "ckpt"
    full = fit(
        adapter,
        prompts,
        source_layers=[0, 2],
        dim_batch=4,
        max_seq_len=64,
        checkpoint_path=checkpoint,
        checkpoint_every=1,
    )
    resumed = fit(
        adapter,
        prompts,
        source_layers=[0, 2],
        dim_batch=4,
        max_seq_len=64,
        checkpoint_path=checkpoint,
        checkpoint_every=1,
    )
    assert resumed.metadata.number_of_prompts == full.metadata.number_of_prompts == 3
    for layer in (0, 2):
        torch.testing.assert_close(resumed.jacobians[layer], full.jacobians[layer])


def test_fit_resume_after_skip_does_not_double_count(tmp_path):
    """A skipped (too-short) prompt must not desync resume indices."""
    adapter = TinyAdapter()
    prompts = [LONG_A, "x", LONG_B]  # "x" tokenizes to 2 tokens -> skipped
    checkpoint = tmp_path / "ckpt"
    reference = fit(adapter, prompts, source_layers=[0, 2], dim_batch=4, max_seq_len=64)
    assert reference.metadata.number_of_prompts == 2
    fit(
        adapter,
        prompts,
        source_layers=[0, 2],
        dim_batch=4,
        max_seq_len=64,
        checkpoint_path=checkpoint,
        checkpoint_every=1,
    )
    resumed = fit(
        adapter,
        prompts,
        source_layers=[0, 2],
        dim_batch=4,
        max_seq_len=64,
        checkpoint_path=checkpoint,
        checkpoint_every=1,
    )
    assert resumed.metadata.number_of_prompts == 2
    for layer in (0, 2):
        torch.testing.assert_close(resumed.jacobians[layer], reference.jacobians[layer])


def test_fit_resume_mismatched_settings_rejected(tmp_path):
    adapter = TinyAdapter()
    checkpoint = tmp_path / "ckpt"
    fit(
        adapter,
        [LONG_A],
        source_layers=[0, 1],
        dim_batch=4,
        max_seq_len=64,
        checkpoint_path=checkpoint,
    )
    with pytest.raises(ValueError, match="source_layers"):
        fit(
            adapter,
            [LONG_A],
            source_layers=[0, 2],
            dim_batch=4,
            max_seq_len=64,
            checkpoint_path=checkpoint,
        )


def test_fit_cancellation(tmp_path):
    adapter = TinyAdapter()
    calls = {"n": 0}

    def cancel_after_first() -> bool:
        calls["n"] += 1
        return calls["n"] > 1

    with pytest.raises(FittingCancelled):
        fit(
            adapter,
            [LONG_A, LONG_B, LONG_C],
            source_layers=[0],
            dim_batch=4,
            max_seq_len=64,
            checkpoint_path=tmp_path / "ckpt",
            should_cancel=cancel_after_first,
        )
    # Cancellation wrote a resumable checkpoint.
    resumed = fit(
        adapter,
        [LONG_A, LONG_B, LONG_C],
        source_layers=[0],
        dim_batch=4,
        max_seq_len=64,
        checkpoint_path=tmp_path / "ckpt",
    )
    assert resumed.metadata.number_of_prompts == 3


def test_negative_layer_indices_normalized():
    adapter = TinyAdapter()
    prompt = "the quick brown fox " * 4
    j_neg, _, _ = jacobian_for_prompt(
        adapter, prompt, source_layers=[-4, -3], target_layer=-1, dim_batch=4, max_seq_len=64
    )
    j_pos, _, _ = jacobian_for_prompt(
        adapter, prompt, source_layers=[0, 1], target_layer=3, dim_batch=4, max_seq_len=64
    )
    assert set(j_neg) == {0, 1}
    for layer in (0, 1):
        torch.testing.assert_close(j_neg[layer], j_pos[layer])


def test_out_of_range_layers_rejected():
    adapter = TinyAdapter()
    prompt = "the quick brown fox " * 4
    with pytest.raises(ValueError, match="out of range"):
        fit(adapter, [prompt], source_layers=[0, 7], dim_batch=4, max_seq_len=64)
    with pytest.raises(ValueError, match="must all be < target_layer"):
        fit(adapter, [prompt], source_layers=[-1], dim_batch=4, max_seq_len=64)
    with pytest.raises(ValueError, match="target_layer"):
        jacobian_for_prompt(adapter, prompt, source_layers=[0], target_layer=9, dim_batch=4)


def test_nan_gradients_detected():
    adapter = TinyAdapter()
    with torch.no_grad():
        adapter.model.layers[2].linear.weight[0, 0] = float("nan")
    with pytest.raises(RuntimeError, match="non-finite"):
        jacobian_for_prompt(
            adapter, "the quick brown fox " * 4, source_layers=[0], dim_batch=4, max_seq_len=64
        )


def test_merge_weighted_by_prompt_counts():
    """merge() is the prompt-count-weighted mean, not a plain average."""
    adapter = TinyAdapter()
    lens_a = fit(adapter, [LONG_A], source_layers=[0, 1], dim_batch=4, max_seq_len=64)
    lens_b = fit(adapter, [LONG_B, LONG_C], source_layers=[0, 1], dim_batch=4, max_seq_len=64)
    merged = JacobianLens.merge([lens_a, lens_b])
    reference = fit(
        adapter, [LONG_A, LONG_B, LONG_C], source_layers=[0, 1], dim_batch=4, max_seq_len=64
    )
    assert merged.metadata.number_of_prompts == 3
    for layer in (0, 1):
        torch.testing.assert_close(
            merged.jacobians[layer], reference.jacobians[layer], rtol=1e-5, atol=1e-6
        )


def test_merge_rejects_mismatched_shards():
    adapter = TinyAdapter()
    lens_a = fit(adapter, [LONG_A], source_layers=[0], dim_batch=4, max_seq_len=64)
    lens_b = fit(adapter, [LONG_B], source_layers=[1], dim_batch=4, max_seq_len=64)
    with pytest.raises(ValueError, match="disagree"):
        JacobianLens.merge([lens_a, lens_b])
    with pytest.raises(ValueError, match="at least one"):
        JacobianLens.merge([])


def test_final_layer_behavior_transport_recovers_model_output():
    """Transporting the penultimate residual with J_{L-2} = I + W_{L-1}
    exactly reproduces the final-layer residual (the tiny model is linear)."""
    adapter = TinyAdapter(n_layers=4, hidden=8)
    lens = fit(adapter, [LONG_A], source_layers=[2], dim_batch=4, max_seq_len=64)
    inputs = adapter.prepare_inputs("the quick brown fox")
    result = adapter.forward_with_activations(inputs, layers=[2, 3])
    h2 = result.activations[2][0].float()
    h3 = result.activations[3][0].float()
    torch.testing.assert_close(lens.transport(h2, 2), h3, rtol=1e-4, atol=1e-5)


def test_tiny_model_dim_batch_invariance():
    """dim_batch is a memory knob only: results are identical."""
    adapter = TinyAdapter()
    prompt = "the quick brown fox " * 4
    j_small, _, _ = jacobian_for_prompt(adapter, prompt, source_layers=[0], dim_batch=2)
    j_large, _, _ = jacobian_for_prompt(adapter, prompt, source_layers=[0], dim_batch=8)
    torch.testing.assert_close(j_small[0], j_large[0], rtol=1e-5, atol=1e-6)
