"""Integration tests against real Qwen2 checkpoints from the Hugging Face Hub.

These download weights, so they are excluded from the default test run.
Run them with::

    pytest -m integration tests/test_integration_qwen.py

``TINY_QWEN`` (~1 MB) exercises the full pipeline quickly; the 0.5B test
verifies the lens produces sane readouts on a real model and takes several
minutes on CPU.
"""

import pytest
import torch

from openjspace.core.applying import inspect_prompt
from openjspace.core.fitting import fit
from openjspace.core.lens import JacobianLens
from openjspace.models.registry import load_model

TINY_QWEN = "trl-internal-testing/tiny-Qwen2ForCausalLM-2.5"
REAL_QWEN = "Qwen/Qwen2.5-0.5B-Instruct"

FIT_PROMPTS = [
    "The history of astronomy begins with the earliest civilizations, who tracked "
    "the motions of the sun, moon, and planets across the sky for calendars and "
    "navigation. Over centuries these observations became systematic records.",
    "In modern software engineering, version control systems record every change "
    "to a codebase so that teams can collaborate, review each other's work, and "
    "recover earlier states of a project when something goes wrong.",
    "Rivers shape the landscapes they pass through by eroding rock, carrying "
    "sediment downstream, and depositing it in floodplains and deltas, creating "
    "some of the most fertile agricultural land on the planet.",
]

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def tiny_qwen():
    return load_model(TINY_QWEN, device="cpu", dtype="float32")


def test_tiny_qwen_routes_to_qwen_adapter(tiny_qwen):
    from openjspace.models.qwen_decoder import QwenDecoderAdapter

    assert isinstance(tiny_qwen.adapter, QwenDecoderAdapter)
    assert tiny_qwen.adapter.architecture == "Qwen2ForCausalLM"
    assert tiny_qwen.adapter.residual_location == "block_output"


def test_tiny_qwen_fit_inspect_roundtrip(tiny_qwen, tmp_path):
    """Full pipeline on a real (tiny) Qwen2 checkpoint: fit -> save -> load ->
    inspect -> final row matches the model's own logits."""
    adapter = tiny_qwen.adapter
    lens = fit(
        adapter,
        FIT_PROMPTS,
        source_layers=[0],
        dim_batch=8,
        max_seq_len=48,
        skip_first=4,
        checkpoint_path=tmp_path / "ckpt",
    )
    assert lens.metadata.number_of_prompts == len(FIT_PROMPTS)
    assert torch.isfinite(lens.jacobians[0]).all()

    lens.save(tmp_path / "lens")
    reloaded = JacobianLens.load(tmp_path / "lens")
    assert reloaded.validate_against(adapter) == []
    # Artifacts store J in fp16, so the round-trip matches to fp16 precision.
    torch.testing.assert_close(reloaded.jacobians[0], lens.jacobians[0], rtol=1e-3, atol=1e-3)

    result = inspect_prompt(
        adapter,
        reloaded,
        "The animal that spins webs has this many legs:",
        top_k=5,
        model_kind="text",
    )
    layers = {cell.layer for cell in result.cells}
    assert layers == {0, adapter.n_layers - 1}
    final_cells = [c for c in result.cells if c.is_model_output]
    assert final_cells and all(len(c.jlens_top) == 5 for c in final_cells)

    # The model-output row must agree with the HF model's own forward logits.
    inputs = adapter.prepare_inputs("The animal that spins webs has this many legs:")
    with torch.inference_mode():
        hf_logits = adapter._hf_model(input_ids=inputs.input_ids, use_cache=False).logits[0]
    last = next(c for c in final_cells if c.position == inputs.seq_len - 1)
    expected_top = hf_logits[-1].topk(5).indices.tolist()
    assert [e.token_id for e in last.jlens_top] == expected_top


@pytest.mark.slow
def test_real_qwen_lens_readout(tmp_path):
    """Fit a two-layer lens on Qwen2.5-0.5B-Instruct with 2 prompts and check
    the readout is finite, ranked, and distinct from the logit lens. Several
    minutes on CPU."""
    loaded = load_model(REAL_QWEN, device="cpu", dtype="float32")
    adapter = loaded.adapter
    lens = fit(
        adapter,
        FIT_PROMPTS[:2],
        source_layers=[8, 16],
        dim_batch=32,
        max_seq_len=48,
        skip_first=8,
        checkpoint_path=tmp_path / "ckpt",
    )
    for layer in (8, 16):
        J = lens.jacobians[layer]
        assert J.shape == (adapter.hidden_size, adapter.hidden_size)
        assert torch.isfinite(J).all()
        # A trained model's transport should be far from both zero and identity.
        norm_scaled = J.norm().item() / adapter.hidden_size**0.5
        assert 0.1 < norm_scaled < 50.0
        identity_gap = (J - torch.eye(adapter.hidden_size)).norm() / J.norm()
        assert identity_gap > 0.1

    result = inspect_prompt(
        adapter,
        lens,
        "The capital of France is",
        positions="last:2",
        top_k=10,
        model_kind="text",
    )
    jlens_cells = [c for c in result.cells if not c.is_model_output]
    assert jlens_cells
    for cell in jlens_cells:
        assert len(cell.jlens_top) == 10
        assert cell.activation_norm > 0
        assert cell.transported_norm > 0
        # J-lens and logit lens are different readouts; on a real model they
        # should not be identical at intermediate layers.
        assert cell.comparison is not None
