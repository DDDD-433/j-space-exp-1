# Portions derived from anthropics/jacobian-lens (jlens/lens.py)
# Copyright 2026 Anthropic PBC
# Copyright 2026 The OpenJSpace Authors
# SPDX-License-Identifier: Apache-2.0
"""The fitted Jacobian lens: per-layer transport matrices plus metadata.

A :class:`JacobianLens` holds the per-layer ``J_l`` matrices produced by
:func:`openjspace.core.fitting.fit` together with the artifact metadata used
for compatibility validation. The readout is::

    lens_l(h) = W_U · Norm(J_l · h)

where ``Norm`` and ``W_U`` are the model's own final normalization and
unembedding (both applied by the adapter's ``unembed``).
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import torch

from openjspace.core.serialization import (
    LensMetadata,
    check_compatibility,
    load_lens,
    save_lens,
)
from openjspace.models.protocol import LensModelAdapter


class JacobianLens:
    """Per-layer ``J_l`` matrices and their metadata.

    Attributes:
        jacobians: ``{layer_index: Tensor[hidden, hidden]}`` float32. Each
            ``J_l`` maps the residual at layer ``l`` into the final-layer
            (target-layer) basis.
        metadata: The versioned artifact metadata.
    """

    def __init__(self, jacobians: dict[int, torch.Tensor], metadata: LensMetadata) -> None:
        self.jacobians = {layer: J.float() for layer, J in jacobians.items()}
        self.metadata = metadata
        if sorted(self.jacobians) != sorted(metadata.source_layers):
            raise ValueError(
                f"jacobian layers {sorted(self.jacobians)} disagree with metadata "
                f"source_layers {sorted(metadata.source_layers)}"
            )

    @property
    def source_layers(self) -> list[int]:
        return sorted(self.jacobians)

    @property
    def hidden_size(self) -> int:
        return self.metadata.hidden_size

    def __repr__(self) -> str:
        layers = self.source_layers
        return (
            f"JacobianLens(model={self.metadata.model_id!r}, hidden={self.hidden_size}, "
            f"n_prompts={self.metadata.number_of_prompts}, "
            f"layers=[{layers[0]}..{layers[-1]}] ({len(layers)}))"
        )

    def save(self, directory: str | Path) -> Path:
        return save_lens(directory, self.jacobians, self.metadata)

    @classmethod
    def load(cls, directory: str | Path) -> JacobianLens:
        jacobians, metadata = load_lens(directory)
        return cls(jacobians, metadata)

    def validate_against(self, adapter: LensModelAdapter, *, force: bool = False) -> list[str]:
        """Check artifact/model compatibility; see
        :func:`openjspace.core.serialization.check_compatibility`."""
        return check_compatibility(
            self.metadata,
            model_id=adapter.model_id,
            architecture=adapter.architecture,
            hidden_size=adapter.hidden_size,
            vocab_size=adapter.vocab_size,
            n_layers=adapter.n_layers,
            tokenizer_id=adapter.tokenizer_id,
            residual_location=adapter.residual_location,
            model_revision=adapter.model_revision,
            force=force,
        )

    def transport(self, residual: torch.Tensor, layer: int) -> torch.Tensor:
        """Map a residual at ``layer`` into the target-layer basis: ``J_l @ h``.

        Args:
            residual: Tensor of shape ``[..., hidden]``.
            layer: Source layer index (must be in :attr:`source_layers`).
        """
        if layer not in self.jacobians:
            raise ValueError(f"layer {layer} not fitted; fitted layers are {self.source_layers}")
        J = self.jacobians[layer].to(residual.device, residual.dtype)
        return residual @ J.T

    @classmethod
    def merge(cls, lenses: Sequence[JacobianLens]) -> JacobianLens:
        """Combine lenses fitted on disjoint prompt shards.

        The merge weights each shard by its accumulated prompt count
        (``number_of_prompts``) — the outer expectation of the estimator is
        over prompts, so this reproduces the mean a single fit over the union
        of the shards would produce. Shards must agree on model identity,
        shapes, source layers, residual location, and fitting hyperparameters.

        Raises:
            ValueError: If ``lenses`` is empty or shards are incompatible.
        """
        if not lenses:
            raise ValueError("merge() needs at least one lens")
        first = lenses[0].metadata
        for other in lenses[1:]:
            m = other.metadata
            mismatches = [
                name
                for name, a, b in [
                    ("model_id", first.model_id, m.model_id),
                    ("hidden_size", first.hidden_size, m.hidden_size),
                    ("vocab_size", first.vocab_size, m.vocab_size),
                    ("n_layers", first.n_layers, m.n_layers),
                    ("source_layers", first.source_layers, m.source_layers),
                    ("target_layer", first.target_layer, m.target_layer),
                    ("residual_location", first.residual_location, m.residual_location),
                    ("skip_first_positions", first.skip_first_positions, m.skip_first_positions),
                    ("sequence_length", first.sequence_length, m.sequence_length),
                ]
                if a != b
            ]
            if mismatches:
                raise ValueError(f"lens shards disagree on: {', '.join(mismatches)}")
        weights = [lens.metadata.number_of_prompts for lens in lenses]
        if any(w <= 0 for w in weights):
            raise ValueError("every shard must have number_of_prompts > 0 to merge")
        total = sum(weights)
        merged: dict[int, torch.Tensor] = {}
        for layer in first.source_layers:
            acc = torch.zeros_like(lenses[0].jacobians[layer])
            for lens, weight in zip(lenses, weights, strict=True):
                acc += lens.jacobians[layer] * weight
            merged[layer] = acc / total
        metadata = first.model_copy(
            update={
                "number_of_prompts": total,
                "number_of_valid_positions": sum(
                    lens.metadata.number_of_valid_positions for lens in lenses
                ),
                "notes": (first.notes + " " if first.notes else "")
                + f"[merged from {len(lenses)} shards]",
            }
        )
        return cls(merged, metadata)
