# Portions derived from anthropics/jacobian-lens (tests/tiny.py)
# Copyright 2026 Anthropic PBC
# Copyright 2026 The OpenJSpace Authors
# SPDX-License-Identifier: Apache-2.0
"""A tiny CPU-only decoder + adapter for tests, CI, and demos.

Implements the full :class:`~openjspace.models.protocol.LensModelAdapter`
protocol without downloading any weights. Residual blocks are
``h + 0.1 * linear(h)``: the small gain keeps the Jacobian well-conditioned so
the late-layer ``diag(J) ~= 1`` property holds, and ``J_{L-2} = I + W_{L-1}``
exactly, which the mathematical tests exploit.
"""

from __future__ import annotations

from collections.abc import Sequence

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

TINY_MODEL_ID = "openjspace/tiny-test-model"


class _ResidualBlock(nn.Module):
    def __init__(self, hidden: int) -> None:
        super().__init__()
        self.linear = nn.Linear(hidden, hidden, bias=False)
        with torch.no_grad():
            self.linear.weight.mul_(0.1)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return hidden + self.linear(hidden)


class ByteTokenizer:
    """Toy tokenizer mapping bytes into a small vocabulary."""

    bos_token_id = 0

    def __init__(self, vocab_size: int = 32) -> None:
        self.vocab_size = vocab_size

    def encode(self, text: str, *, max_length: int = 128) -> list[int]:
        span = self.vocab_size - 2
        return [self.bos_token_id, *[1 + b % span for b in text.encode()][: max_length - 1]]

    def decode_one(self, token_id: int) -> str:
        if token_id == self.bos_token_id:
            return "<bos>"
        return chr(96 + int(token_id))


class TinyModel(nn.Module):
    """``n_layers``-layer residual stack on CPU (default ``hidden=8``, ``vocab=32``)."""

    def __init__(
        self, n_layers: int = 4, hidden: int = 8, vocab_size: int = 32, seed: int = 0
    ) -> None:
        super().__init__()
        torch.manual_seed(seed)
        self.n_layers = n_layers
        self.hidden = hidden
        self.vocab_size = vocab_size
        self.embed_tokens = nn.Embedding(vocab_size, hidden)
        self.layers = nn.ModuleList(_ResidualBlock(hidden) for _ in range(n_layers))
        self.norm = nn.LayerNorm(hidden)
        self.lm_head = nn.Linear(hidden, vocab_size, bias=False)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        hidden = self.embed_tokens(input_ids)
        for block in self.layers:
            hidden = block(hidden)
        return hidden


class TinyAdapter:
    """:class:`LensModelAdapter` over :class:`TinyModel`; used by unit tests."""

    residual_location: ResidualLocation = "block_output"

    def __init__(self, model: TinyModel | None = None, **model_kwargs: int) -> None:
        self.model = model if model is not None else TinyModel(**model_kwargs)
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad_(False)
        self.tokenizer = ByteTokenizer(self.model.vocab_size)
        self.model_id = TINY_MODEL_ID
        self.model_revision: str | None = None
        self.architecture = "TinyModel"
        self.n_layers = self.model.n_layers
        self.hidden_size = self.model.hidden
        self.vocab_size = self.model.vocab_size
        self.tokenizer_id = "openjspace/tiny-byte-tokenizer"
        self.device = "cpu"

    def tokenize_text(self, text: str, *, max_length: int = 512) -> TokenizedInput:
        ids = self.tokenizer.encode(text, max_length=max_length)
        return TokenizedInput(
            input_ids=torch.tensor([ids], dtype=torch.long),
            token_strings=[self.tokenizer.decode_one(t) for t in ids],
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
            raise ValueError("the tiny test model does not accept images")
        tokenized = self.tokenize_text(prompt, max_length=max_length)
        positions = [
            PositionMetadata(
                index=i,
                modality="special" if token_id == self.tokenizer.bos_token_id else "text",
                token_id=int(token_id),
                token_text=text,
            )
            for i, (token_id, text) in enumerate(
                zip(tokenized.input_ids[0].tolist(), tokenized.token_strings, strict=True)
            )
        ]
        return ModelInputs(input_ids=tokenized.input_ids, positions=positions)

    def get_residual_modules(self) -> Sequence[nn.Module]:
        return list(self.model.layers)

    def get_final_norm(self) -> nn.Module:
        return self.model.norm

    def get_unembedding_weight(self) -> torch.Tensor:
        return self.model.lm_head.weight

    def decode_token_ids(self, token_ids: Sequence[int]) -> list[str]:
        return [self.tokenizer.decode_one(t) for t in token_ids]

    def classify_positions(self, inputs: ModelInputs) -> list[PositionMetadata]:
        return inputs.positions

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
        with ActivationRecorder(self.model.layers, at=layers, start_graph_at=grad_from) as rec:
            if grad_from is None:
                with torch.inference_mode():
                    self.model(input_ids)
            else:
                self.model(input_ids)
            activations = dict(rec.activations)
        return ForwardResult(activations=activations)

    def unembed(self, residual: torch.Tensor) -> torch.Tensor:
        return self.model.lm_head(self.model.norm(residual.float()))
