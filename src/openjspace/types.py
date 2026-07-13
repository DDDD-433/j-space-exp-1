"""Shared typed structures used across the OpenJSpace package."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import torch

ResidualLocation = Literal[
    "block_input",
    "block_output",
    "post_attention",
    "post_mlp",
]
"""Which tensor an adapter captures as "the residual stream" at a layer.

Every adapter uses one consistent location across all layers, and the location
is recorded in lens metadata so incompatible lenses cannot be silently applied.
v0.1 adapters all use ``block_output``.
"""

Modality = Literal["text", "image_token", "image_boundary", "special", "unknown"]

PatchMappingStatus = Literal["exact", "approximate", "unavailable"]
"""How faithfully decoder image-token positions map back to image patch geometry.

- ``exact``: a one-to-one position -> (row, col) patch mapping exists.
- ``approximate``: positions correspond to merged/shuffled patch groups; the
  reported (row, col) is a group location, not a single vision-encoder patch.
- ``unavailable``: the architecture resamples visual tokens (no spatial map);
  no coordinates are reported and none should be invented.
"""


@dataclass
class PositionMetadata:
    """Modality and provenance of one decoder sequence position."""

    index: int
    modality: Modality
    token_id: int | None = None
    token_text: str | None = None
    image_index: int | None = None
    patch_index: int | None = None
    patch_row: int | None = None
    patch_col: int | None = None


@dataclass
class TokenizedInput:
    """A tokenized text prompt: ``input_ids`` of shape ``[1, seq_len]``."""

    input_ids: torch.Tensor
    token_strings: list[str]

    @property
    def seq_len(self) -> int:
        return int(self.input_ids.shape[1])


@dataclass
class ModelInputs:
    """Exact inputs for one forward pass, plus position metadata.

    ``extra`` carries adapter-specific tensors (e.g. ``pixel_values``,
    ``attention_mask``) passed through to the underlying model verbatim so the
    multimodal sequence is preserved exactly as the processor built it.
    """

    input_ids: torch.Tensor  # [1, seq_len]
    positions: list[PositionMetadata]
    extra: dict[str, torch.Tensor] = field(default_factory=dict)
    patch_mapping: PatchMappingStatus = "unavailable"
    patch_grids: list[tuple[int, int]] = field(default_factory=list)
    """(rows, cols) of the decoder-visible token grid per image, when known."""

    @property
    def seq_len(self) -> int:
        return int(self.input_ids.shape[1])


@dataclass
class ForwardResult:
    """Activations recorded during one forward pass.

    ``activations`` maps layer index to the residual tensor at the adapter's
    declared residual location, shape ``[batch, seq_len, hidden]``. Tensors are
    differentiable when the pass was run with gradients enabled.
    """

    activations: dict[int, torch.Tensor]
    final_logits: torch.Tensor | None = None  # [batch, seq_len, vocab]
