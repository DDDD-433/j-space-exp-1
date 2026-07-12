"""Fast unit tests for VLM modality classification and patch geometry.

These exercise the pure classification/geometry logic of
:class:`~openjspace.models.smolvlm.SmolVLMAdapter` without loading any model
weights, by constructing an adapter shell with ``object.__new__`` and setting
only the fields the logic reads. Real-model behaviour is covered by the
integration tests in ``tests/test_integration_smolvlm.py``.
"""

import pytest

from openjspace.analysis.modality_map import (
    assign_patch_grid,
    image_token_positions,
    modality_counts,
)
from openjspace.models.smolvlm import SmolVLMAdapter
from openjspace.types import ModelInputs, PositionMetadata

IMAGE_TOKEN_ID = 900
BOUNDARY_IDS = {800, 801}  # fake_token_around_image, row/col markers
SPECIAL_IDS = {0, 1}  # bos / eos


def _shell() -> SmolVLMAdapter:
    """A SmolVLMAdapter with only the attributes the classifier reads."""
    adapter = object.__new__(SmolVLMAdapter)
    adapter._image_token_id = IMAGE_TOKEN_ID
    adapter._boundary_ids = BOUNDARY_IDS
    adapter._special_ids = SPECIAL_IDS
    adapter._scale_factor = 4
    return adapter


def _seq(ids: list[int]) -> tuple[list[int], list[str]]:
    return ids, [f"tok{i}" for i in ids]


def test_text_positions_classified():
    adapter = _shell()
    ids, strings = _seq([1, 50, 51, 52, 0])  # special, text, text, text, special
    positions = adapter._classify(ids, strings)
    assert [p.modality for p in positions] == [
        "special",
        "text",
        "text",
        "text",
        "special",
    ]


def test_image_positions_classified():
    adapter = _shell()
    # boundary, 4 image tokens, boundary, text
    ids, strings = _seq([800, 900, 900, 900, 900, 801, 50])
    positions = adapter._classify(ids, strings)
    assert [p.modality for p in positions] == [
        "image_boundary",
        "image_token",
        "image_token",
        "image_token",
        "image_token",
        "image_boundary",
        "text",
    ]


def test_multiple_images_distinguishable():
    adapter = _shell()
    # image A (4 tokens), text, image B (4 tokens)
    ids, strings = _seq([800, 900, 900, 900, 900, 50, 800, 900, 900, 900, 900])
    positions = adapter._classify(ids, strings)
    grids = adapter._assign_patch_geometry(positions)
    assert grids == [(2, 2), (2, 2)]
    imgtoks = [p for p in positions if p.modality == "image_token"]
    assert {p.image_index for p in imgtoks} == {0, 1}
    assert [p.image_index for p in imgtoks] == [0, 0, 0, 0, 1, 1, 1, 1]


def test_square_run_gets_row_col_geometry():
    adapter = _shell()
    ids, strings = _seq([900] * 9)  # 3x3
    positions = adapter._classify(ids, strings)
    grids = adapter._assign_patch_geometry(positions)
    assert grids == [(3, 3)]
    coords = [(p.patch_row, p.patch_col) for p in positions]
    assert coords[0] == (0, 0)
    assert coords[4] == (1, 1)
    assert coords[8] == (2, 2)


def test_non_square_run_reports_strip_without_coords():
    adapter = _shell()
    ids, strings = _seq([900] * 5)  # not a perfect square
    positions = adapter._classify(ids, strings)
    grids = adapter._assign_patch_geometry(positions)
    assert grids == [(1, 5)]
    # No fabricated 2D coordinates for a non-square strip.
    assert all(p.patch_row is None and p.patch_col is None for p in positions)
    assert [p.patch_index for p in positions] == [0, 1, 2, 3, 4]


def test_modality_counts_and_filtering():
    positions = [
        PositionMetadata(index=0, modality="special", token_id=1),
        PositionMetadata(index=1, modality="image_token", token_id=900, image_index=0),
        PositionMetadata(index=2, modality="image_token", token_id=900, image_index=0),
        PositionMetadata(index=3, modality="text", token_id=50),
    ]
    assert modality_counts(positions) == {"special": 1, "image_token": 2, "text": 1}
    assert len(image_token_positions(positions, image_index=0)) == 2
    assert len(image_token_positions(positions, image_index=1)) == 0


def test_assign_patch_grid_rejects_mismatch():
    positions = [
        PositionMetadata(index=i, modality="image_token", token_id=900, image_index=0)
        for i in range(4)
    ]
    with pytest.raises(ValueError, match="!= grid"):
        assign_patch_grid(positions, image_index=0, grid_rows=3, grid_cols=3)


def test_model_inputs_default_patch_mapping_unavailable():
    """Text-only inputs must report patch mapping as unavailable, never fabricate."""
    inputs = ModelInputs(input_ids=None, positions=[])  # type: ignore[arg-type]
    assert inputs.patch_mapping == "unavailable"
    assert inputs.patch_grids == []
