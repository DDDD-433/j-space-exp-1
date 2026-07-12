"""Helpers for reasoning about position modality in multimodal sequences."""

from __future__ import annotations

from collections import Counter

from openjspace.types import Modality, PositionMetadata


def modality_counts(positions: list[PositionMetadata]) -> dict[str, int]:
    """Number of positions per modality."""
    return dict(Counter(p.modality for p in positions))


def image_token_positions(
    positions: list[PositionMetadata], image_index: int | None = None
) -> list[PositionMetadata]:
    """All image-token positions, optionally restricted to one image."""
    return [
        p
        for p in positions
        if p.modality == "image_token" and (image_index is None or p.image_index == image_index)
    ]


def assign_patch_grid(
    positions: list[PositionMetadata],
    image_index: int,
    grid_rows: int,
    grid_cols: int,
) -> None:
    """Assign row/col coordinates to one image's token positions in reading
    order. Only valid when the number of tokens equals ``rows * cols``; callers
    must have verified the mapping status first (never fabricate geometry).

    Raises:
        ValueError: When the token count does not match the grid.
    """
    tokens = image_token_positions(positions, image_index)
    if len(tokens) != grid_rows * grid_cols:
        raise ValueError(
            f"image {image_index}: {len(tokens)} tokens != grid {grid_rows}x{grid_cols}"
        )
    for i, meta in enumerate(tokens):
        meta.patch_index = i
        meta.patch_row = i // grid_cols
        meta.patch_col = i % grid_cols


def modality_of(positions: list[PositionMetadata], index: int) -> Modality:
    for p in positions:
        if p.index == index:
            return p.modality
    return "unknown"
