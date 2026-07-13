# Portions derived from anthropics/jacobian-lens (jlens/hf.py)
# Copyright 2026 Anthropic PBC
# Copyright 2026 The OpenJSpace Authors
# SPDX-License-Identifier: Apache-2.0
"""Generic Hugging Face causal-decoder adapter.

Wraps an already-loaded HF ``*ForCausalLM`` as a
:class:`~openjspace.models.protocol.LensModelAdapter`. Model loading (device
placement, dtype, revision) is handled by :mod:`openjspace.models.registry`;
this module only locates the residual stack inside whatever it is handed.

The captured residual location is the **block output** (the full residual
stream after each decoder block), consistent across all layers and recorded in
lens metadata.
"""

from __future__ import annotations

import functools
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, cast

import torch
from torch import nn

from openjspace.core.hooks import ActivationRecorder
from openjspace.types import (
    ForwardResult,
    ModelInputs,
    PositionMetadata,
    ResidualLocation,
    TokenizedInput,
)


def _resolve_attr_path(obj: Any, dotted_path: str) -> Any:
    return functools.reduce(getattr, dotted_path.split("."), obj)


@dataclass(frozen=True)
class Layout:
    """Where the lens-relevant submodules live inside a HuggingFace model.

    Attributes:
        path: Dotted attribute path from the ``*ForCausalLM`` to the bare text
            decoder (the module to call for a hooks-visible forward pass).
        layers: Attribute name on the text decoder for the residual blocks.
        norm: Attribute name for the final pre-unembed norm.
        embed: Attribute name for the input token embedding.
        lm_head: Attribute name on the ``*ForCausalLM`` for the unembedding.
    """

    path: str
    layers: str = "layers"
    norm: str = "norm"
    embed: str = "embed_tokens"
    lm_head: str = "lm_head"


#: Known layouts, tried in order. The first whose ``path`` resolves and whose
#: text decoder has all three of ``layers``/``norm``/``embed`` wins. Covers
#: Llama / Qwen / Mistral / Gemma / OLMo / StableLM (the modern HF default),
#: their multimodal-wrapper variants, plus Phi, GPT-2, and GPT-NeoX.
LAYOUTS: tuple[Layout, ...] = (
    Layout("model"),
    Layout("model.language_model"),
    Layout("language_model"),
    Layout("model", norm="final_layernorm"),  # Phi
    Layout("transformer", layers="h", norm="ln_f", embed="wte"),  # GPT-2
    Layout("gpt_neox", norm="final_layer_norm", embed="embed_in", lm_head="embed_out"),  # Pythia
)


class UnsupportedArchitectureError(ValueError):
    """The model's internal layout could not be located."""


def find_layout(hf_model: nn.Module) -> Layout:
    """Locate the text decoder inside an HF ``*ForCausalLM`` /
    ``*ForConditionalGeneration`` by trying :data:`LAYOUTS` in order.

    Raises:
        UnsupportedArchitectureError: With an actionable message when no known
            layout matches.
    """
    for layout in LAYOUTS:
        try:
            candidate = _resolve_attr_path(hf_model, layout.path)
        except AttributeError:
            continue
        if all(
            hasattr(candidate, attr) for attr in (layout.layers, layout.norm, layout.embed)
        ) and hasattr(hf_model, layout.lm_head):
            return layout
    raise UnsupportedArchitectureError(
        f"could not locate the text decoder inside {type(hf_model).__name__} "
        f"(tried {len(LAYOUTS)} known layouts). Pass an explicit Layout to "
        f"HFDecoderAdapter(layout=...), or open an issue with the model id."
    )


class HFDecoderAdapter:
    """:class:`LensModelAdapter` over a loaded HuggingFace causal LM.

    Holds references into the caller's model; nothing is copied. The
    constructor mutates that model in place: it is put in eval mode and every
    parameter gets ``requires_grad_(False)`` (the Jacobian fit needs gradients
    only with respect to activations).
    """

    residual_location: ResidualLocation = "block_output"

    def __init__(
        self,
        hf_model: nn.Module,
        tokenizer: Any,
        *,
        model_id: str = "",
        model_revision: str | None = None,
        layout: Layout | None = None,
        force_bos: bool = True,
    ) -> None:
        self._hf_model = hf_model
        self.tokenizer = tokenizer
        if (
            force_bos
            and getattr(tokenizer, "bos_token_id", None) is not None
            and hasattr(tokenizer, "add_bos_token")
        ):
            # Some instruction-tuned checkpoints ship add_bos_token=False;
            # raw-text prompts degrade without an attention-sink BOS.
            tokenizer.add_bos_token = True

        hf_model.eval()
        for param in hf_model.parameters():
            param.requires_grad_(False)

        self.layout = layout if layout is not None else find_layout(hf_model)
        self._text_module = _resolve_attr_path(hf_model, self.layout.path)
        self._layers: Sequence[nn.Module] = getattr(self._text_module, self.layout.layers)
        self._final_norm: nn.Module = getattr(self._text_module, self.layout.norm)
        self._embed_tokens: Any = getattr(self._text_module, self.layout.embed)
        self._lm_head: Any = getattr(hf_model, self.layout.lm_head)

        text_config = cast(Any, hf_model).config.get_text_config()
        self.model_id = model_id or getattr(hf_model.config, "_name_or_path", "") or "unknown"
        self.model_revision = model_revision
        self.architecture = type(hf_model).__name__
        self.n_layers: int = text_config.num_hidden_layers
        self.hidden_size: int = text_config.hidden_size
        self.vocab_size: int = int(self._lm_head.weight.shape[0])
        self.tokenizer_id = getattr(tokenizer, "name_or_path", "") or self.model_id
        self._logit_softcap: float | None = getattr(text_config, "final_logit_softcapping", None)
        if len(self._layers) != self.n_layers:
            raise UnsupportedArchitectureError(
                f"config.num_hidden_layers={self.n_layers} but found "
                f"{len(self._layers)} blocks at {self.layout.path}.{self.layout.layers}"
            )

    def __repr__(self) -> str:
        return (
            f"HFDecoderAdapter({self.architecture}, n_layers={self.n_layers}, "
            f"hidden={self.hidden_size}, vocab={self.vocab_size})"
        )

    @property
    def device(self) -> str:  # type: ignore[override]
        return str(self._embed_tokens.weight.device)

    @property
    def torch_dtype(self) -> torch.dtype:
        return self._embed_tokens.weight.dtype

    # ------------------------------------------------------------------ #
    # Tokenization and inputs
    # ------------------------------------------------------------------ #

    def tokenize_text(self, text: str, *, max_length: int = 512) -> TokenizedInput:
        encoded = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length)
        ids = encoded.input_ids
        return TokenizedInput(
            input_ids=ids.to(self._embed_tokens.weight.device),
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
        if images:
            raise ValueError(f"{self.architecture} is a text-only model and does not accept images")
        if use_chat_template:
            prompt = self.tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False,
                add_generation_prompt=True,
            )
        tokenized = self.tokenize_text(prompt, max_length=max_length)
        special_ids = set(getattr(self.tokenizer, "all_special_ids", []) or [])
        positions = [
            PositionMetadata(
                index=i,
                modality="special" if token_id in special_ids else "text",
                token_id=int(token_id),
                token_text=text,
            )
            for i, (token_id, text) in enumerate(
                zip(tokenized.input_ids[0].tolist(), tokenized.token_strings, strict=True)
            )
        ]
        return ModelInputs(input_ids=tokenized.input_ids, positions=positions)

    # ------------------------------------------------------------------ #
    # Structure accessors
    # ------------------------------------------------------------------ #

    def get_residual_modules(self) -> Sequence[nn.Module]:
        return self._layers

    def get_final_norm(self) -> nn.Module:
        return self._final_norm

    def get_unembedding_weight(self) -> torch.Tensor:
        return self._lm_head.weight

    def decode_token_ids(self, token_ids: Sequence[int]) -> list[str]:
        return [
            self.tokenizer.decode([int(t)], clean_up_tokenization_spaces=False) for t in token_ids
        ]

    def classify_positions(self, inputs: ModelInputs) -> list[PositionMetadata]:
        return inputs.positions

    # ------------------------------------------------------------------ #
    # Forward and readout
    # ------------------------------------------------------------------ #

    def _forward(self, input_ids: torch.Tensor, extra: dict[str, torch.Tensor]) -> None:
        self._text_module(input_ids=input_ids, use_cache=False, **extra)

    def forward_with_activations(
        self,
        inputs: ModelInputs,
        layers: Sequence[int],
        *,
        grad_from: int | None = None,
        replicate_batch: int = 1,
    ) -> ForwardResult:
        input_ids = inputs.input_ids
        if replicate_batch > 1:
            input_ids = input_ids.expand(replicate_batch, -1)
        with ActivationRecorder(self._layers, at=layers, start_graph_at=grad_from) as rec:
            if grad_from is None:
                with torch.inference_mode():
                    self._forward(input_ids, inputs.extra)
            else:
                self._forward(input_ids, inputs.extra)
            activations = dict(rec.activations)
        return ForwardResult(activations=activations)

    def unembed(self, residual: torch.Tensor) -> torch.Tensor:
        weight = self._lm_head.weight
        normed = self._final_norm(residual.to(weight.dtype).to(weight.device))
        logits = self._lm_head(normed)
        if self._logit_softcap is not None:
            logits = self._logit_softcap * torch.tanh(logits / self._logit_softcap)
        return logits
