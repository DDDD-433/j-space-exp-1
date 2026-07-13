"""Integration tests for the SmolVLM (Idefics3) VLM adapter.

Downloads the 256M SmolVLM checkpoint and installs ``torchvision`` (needed by
the Idefics3 image processor). Run with::

    pytest -m integration tests/test_integration_smolvlm.py

Image splitting is disabled (``do_image_splitting=False``) so an image expands
to a single 8x8 tile (~64 image tokens), keeping the CPU forward pass fast.
"""

import numpy as np
import pytest

pytestmark = pytest.mark.integration

SMOLVLM = "HuggingFaceTB/SmolVLM-256M-Instruct"


def _synthetic_image(size: int = 48):
    from PIL import Image

    rng = np.random.default_rng(0)
    return Image.fromarray((rng.random((size, size, 3)) * 255).astype("uint8"))


@pytest.fixture(scope="module")
def smolvlm():
    from openjspace.models.registry import load_model

    loaded = load_model(SMOLVLM, device="cpu", dtype="float32")
    loaded.adapter.do_image_splitting = False
    return loaded


def test_smolvlm_routed_as_vlm(smolvlm):
    from openjspace.models.smolvlm import SmolVLMAdapter

    assert smolvlm.kind == "vlm"
    assert isinstance(smolvlm.adapter, SmolVLMAdapter)
    assert smolvlm.adapter.residual_location == "block_output"
    assert smolvlm.adapter.n_layers == 30
    assert smolvlm.adapter.hidden_size == 576


def test_smolvlm_image_positions_classified(smolvlm):
    from collections import Counter

    adapter = smolvlm.adapter
    inputs = adapter.prepare_inputs(
        "What is in this image?", images=[_synthetic_image()], max_length=4096
    )
    counts = Counter(p.modality for p in inputs.positions)
    assert counts["image_token"] == 64  # 8x8 single tile, no splitting
    assert counts["text"] > 0
    assert counts["image_boundary"] > 0
    assert inputs.patch_mapping == "approximate"
    assert inputs.patch_grids == [(8, 8)]
    imgtoks = [p for p in inputs.positions if p.modality == "image_token"]
    assert all(p.image_index == 0 for p in imgtoks)
    assert imgtoks[0].patch_row == 0 and imgtoks[0].patch_col == 0
    assert imgtoks[-1].patch_row == 7 and imgtoks[-1].patch_col == 7


def test_smolvlm_multiple_images_distinguishable(smolvlm):
    adapter = smolvlm.adapter
    inputs = adapter.prepare_inputs(
        "Compare these images.",
        images=[_synthetic_image(), _synthetic_image(64)],
        max_length=4096,
    )
    imgtoks = [p for p in inputs.positions if p.modality == "image_token"]
    assert {p.image_index for p in imgtoks} == {0, 1}


def test_smolvlm_text_only_forward(smolvlm):
    """The VLM decoder must also run text-only (used for text lens fitting)."""
    adapter = smolvlm.adapter
    inputs = adapter.prepare_inputs("The capital of France is")
    assert inputs.patch_mapping == "unavailable"
    assert all(p.modality in ("text", "special") for p in inputs.positions)
    result = adapter.forward_with_activations(inputs, layers=[adapter.n_layers - 1])
    assert result.activations[adapter.n_layers - 1].shape[-1] == adapter.hidden_size


def test_smolvlm_image_inspection_produces_run(smolvlm, tmp_path):
    """An image upload produces a valid inspection result whose image-token
    cells can be read out across layers."""
    from openjspace.core.applying import inspect_prompt
    from openjspace.core.fitting import fit
    from openjspace.report.schema import RunResult

    adapter = smolvlm.adapter
    # Fit a tiny 1-layer lens (text-only) just to have a compatible artifact.
    lens = fit(
        adapter,
        [
            "A photograph usually shows objects, people, or scenery arranged "
            "within a rectangular frame with light and shadow.",
        ],
        source_layers=[10],
        dim_batch=64,
        max_seq_len=32,
        skip_first=4,
    )
    inputs = adapter.prepare_inputs(
        "What is in this image?", images=[_synthetic_image()], max_length=4096
    )
    result = inspect_prompt(
        adapter,
        lens,
        "What is in this image?",
        images=[_synthetic_image()],
        prepared_inputs=inputs,
        layers=[10],
        positions="all",
        top_k=5,
        model_kind="vlm",
    )
    assert result.metadata.patch_mapping == "approximate"
    assert any(w for w in result.metadata.warnings if "patch mapping" in w)
    image_positions = [p for p in result.positions if p.modality == "image_token"]
    assert len(image_positions) == 64
    # There is a lens readout at layer 10 for an image-token position.
    an_image_index = image_positions[0].index
    cell = result.cell(10, an_image_index)
    assert cell is not None and len(cell.jlens_top) == 5

    # Round-trips through JSON.
    path = tmp_path / "run.json"
    path.write_text(result.model_dump_json(), encoding="utf-8")
    reloaded = RunResult.model_validate_json(path.read_text())
    assert reloaded.metadata.model_kind == "vlm"
