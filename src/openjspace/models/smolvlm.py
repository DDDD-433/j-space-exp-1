# Portions derived from anthropics/jacobian-lens (jlens/hf.py)
# Copyright 2026 Anthropic PBC
# Copyright 2026 The OpenJSpace Authors
# SPDX-License-Identifier: Apache-2.0
"""SmolVLM / SmolVLM2 (Idefics3) vision-language adapter.

The lightweight VLM used for the first VLM release. We inspect only the
**language decoder** residual stream — the same ``block_output`` location as
the text adapters — *after* the projected visual embeddings have entered the
language-model sequence. The J-lens vocabulary is therefore the text decoder's
vocabulary, and an image-position readout answers:

    "Which text concepts is this visual-token activation disposed to influence
    the decoder to verbalize?"

It does **not** decode the vision encoder's full representation.

Sequence layout (per the official processor + chat template): each image is
split into tiles (a grid of crops plus one global thumbnail); every tile emits
a run of ``<image>`` placeholder tokens (``grid*grid`` of them after the
pixel-shuffle connector), bracketed by ``<fake_token_around_image>``,
``<row_R_col_C>`` and ``<global-img>`` markers. Because the connector shuffles
a ``scale_factor**2`` block of vision patches into a single decoder token,
patch geometry is **approximate**: a decoder image token maps to a *group* of
vision patches, not one pixel patch. We surface that status and never fabricate
a one-to-one pixel mapping.
"""

from __future__ import annotations

import math
import re
from collections.abc import Sequence
from typing import Any, cast

import torch
from torch import nn

from openjspace.core.hooks import ActivationRecorder
from openjspace.models.hf_decoder import HFDecoderAdapter, Layout
from openjspace.types import (
    ForwardResult,
    Modality,
    ModelInputs,
    PositionMetadata,
    TokenizedInput,
)

#: Marker tokens that delimit image tiles; classified as ``image_boundary``.
_BOUNDARY_RE = re.compile(r"fake_token_around_image|global-img|row_\d+_col_\d+")

#: text_model lives under ``model.text_model``; ``lm_head`` is on the root.
SMOLVLM_LAYOUT = Layout(path="model.text_model")


class SmolVLMAdapter(HFDecoderAdapter):
    """:class:`LensModelAdapter` over an Idefics3/SmolVLM model.

    Hooks the language decoder's blocks but runs the *full* multimodal forward
    so projected visual embeddings are merged into the sequence exactly as the
    model sees them.
    """

    def __init__(
        self,
        hf_model: nn.Module,
        processor: Any,
        *,
        model_id: str = "",
        model_revision: str | None = None,
    ) -> None:
        super().__init__(
            hf_model,
            processor.tokenizer,
            model_id=model_id,
            model_revision=model_revision,
            layout=SMOLVLM_LAYOUT,
            force_bos=False,
        )
        self.processor = processor
        #: When False, the processor emits only the global thumbnail tile (one
        #: image-token run) instead of the full crop grid. The default keeps
        #: the model's faithful splitting; tests/interactive CPU use may set
        #: this False to make forward passes tractable.
        self.do_image_splitting: bool = True
        config = cast(Any, hf_model).config
        self._image_token_id = int(config.image_token_id)
        self._scale_factor = int(getattr(config, "scale_factor", 1) or 1)
        # Ids of the tile-delimiter markers (row/col/global/fake).
        added = getattr(processor.tokenizer, "added_tokens_encoder", {}) or {}
        self._boundary_ids = {tid for tok, tid in added.items() if _BOUNDARY_RE.search(tok)}
        self._special_ids = set(getattr(processor.tokenizer, "all_special_ids", []) or [])

    # ------------------------------------------------------------------ #
    # Inputs
    # ------------------------------------------------------------------ #

    def tokenize_text(self, text: str, *, max_length: int = 512) -> TokenizedInput:
        encoded = self.processor.tokenizer(
            text, return_tensors="pt", truncation=True, max_length=max_length
        )
        ids = encoded.input_ids
        return TokenizedInput(
            input_ids=ids.to(self.device),
            token_strings=self.decode_token_ids(ids[0].tolist()),
        )

    def prepare_inputs(
        self,
        prompt: str,
        *,
        images: Sequence[object] | None = None,
        max_length: int = 512,
        use_chat_template: bool = False,
    ) -> ModelInputs:
        """Build the exact multimodal inputs via the official processor.

        With no images this is a plain text forward through the VLM decoder
        (used for text-only lens fitting). With images, the processor's chat
        template inserts the tile markers and ``<image>`` placeholders; we keep
        ``pixel_values``/``pixel_attention_mask`` in ``extra`` verbatim.
        """
        if not images:
            tokenized = self.tokenize_text(prompt, max_length=max_length)
            positions = self._classify(tokenized.input_ids[0].tolist(), tokenized.token_strings)
            return ModelInputs(input_ids=tokenized.input_ids, positions=positions)

        content = [{"type": "image"} for _ in images] + [{"type": "text", "text": prompt}]
        text = self.processor.apply_chat_template(
            [{"role": "user", "content": content}], add_generation_prompt=True
        )
        encoded = self.processor(
            text=text,
            images=list(images),
            return_tensors="pt",
            images_kwargs={"do_image_splitting": self.do_image_splitting},
        )
        input_ids = encoded["input_ids"].to(self.device)
        extra = {
            key: value.to(self.device)
            for key, value in encoded.items()
            if key != "input_ids" and torch.is_tensor(value)
        }
        token_strings = self.decode_token_ids(input_ids[0].tolist())
        positions = self._classify(input_ids[0].tolist(), token_strings)
        grids = self._assign_patch_geometry(positions)
        return ModelInputs(
            input_ids=input_ids,
            positions=positions,
            extra=extra,
            patch_mapping="approximate",
            patch_grids=grids,
        )

    def _classify(self, ids: list[int], strings: list[str]) -> list[PositionMetadata]:
        positions: list[PositionMetadata] = []
        for i, (token_id, text) in enumerate(zip(ids, strings, strict=True)):
            modality: Modality
            if token_id == self._image_token_id:
                modality = "image_token"
            elif token_id in self._boundary_ids:
                modality = "image_boundary"
            elif token_id in self._special_ids:
                modality = "special"
            else:
                modality = "text"
            positions.append(
                PositionMetadata(
                    index=i, modality=modality, token_id=int(token_id), token_text=text
                )
            )
        return positions

    def _assign_patch_geometry(self, positions: list[PositionMetadata]) -> list[tuple[int, int]]:
        """Group consecutive image-token runs into tiles and lay each out on a
        square grid (approximate: each token is a pixel-shuffled patch group).

        Returns the ``(rows, cols)`` grid of each tile in sequence order.
        """
        grids: list[tuple[int, int]] = []
        run: list[PositionMetadata] = []
        image_index = 0

        def flush(run: list[PositionMetadata]) -> None:
            nonlocal image_index
            if not run:
                return
            side = math.isqrt(len(run))
            square = side * side == len(run)
            rows, cols = (side, side) if square else (1, len(run))
            for offset, meta in enumerate(run):
                meta.image_index = image_index
                meta.patch_index = offset
                if square:
                    meta.patch_row = offset // cols
                    meta.patch_col = offset % cols
            grids.append((rows, cols))
            image_index += 1

        for meta in positions:
            if meta.modality == "image_token":
                run.append(meta)
            else:
                flush(run)
                run = []
        flush(run)
        return grids

    def classify_positions(self, inputs: ModelInputs) -> list[PositionMetadata]:
        return inputs.positions

    # ------------------------------------------------------------------ #
    # Forward
    # ------------------------------------------------------------------ #

    def _forward(self, input_ids: torch.Tensor, extra: dict[str, torch.Tensor]) -> None:
        # Run the full multimodal model so vision embeddings are merged into
        # the language sequence; hooks on text_model.layers still fire.
        self._hf_model(input_ids=input_ids, use_cache=False, **extra)

    def forward_with_activations(
        self,
        inputs: ModelInputs,
        layers: Sequence[int],
        *,
        grad_from: int | None = None,
        replicate_batch: int = 1,
    ) -> ForwardResult:
        input_ids = inputs.input_ids
        extra = dict(inputs.extra)
        if replicate_batch > 1:
            if extra:
                raise ValueError(
                    "replicate_batch > 1 is only supported for text-only VLM fitting; "
                    "image inputs cannot be safely replicated along the batch axis"
                )
            input_ids = input_ids.expand(replicate_batch, -1)
        with ActivationRecorder(self._layers, at=layers, start_graph_at=grad_from) as rec:
            if grad_from is None:
                with torch.inference_mode():
                    self._forward(input_ids, extra)
            else:
                self._forward(input_ids, extra)
            activations = dict(rec.activations)
        return ForwardResult(activations=activations)


def load_smolvlm_adapter(
    model_id: str,
    *,
    config: Any,
    device: str,
    torch_dtype: torch.dtype,
    revision: str | None = None,
    trust_remote_code: bool = False,
) -> SmolVLMAdapter:
    """Load a SmolVLM/Idefics3 checkpoint and wrap it in :class:`SmolVLMAdapter`."""
    from transformers import AutoModelForImageTextToText, AutoProcessor

    processor = AutoProcessor.from_pretrained(
        model_id, revision=revision, trust_remote_code=trust_remote_code
    )
    hf_model = cast(
        Any,
        AutoModelForImageTextToText.from_pretrained(
            model_id, revision=revision, dtype=torch_dtype, trust_remote_code=trust_remote_code
        ),
    )
    hf_model.to(device)
    return SmolVLMAdapter(hf_model, processor, model_id=model_id, model_revision=revision)
