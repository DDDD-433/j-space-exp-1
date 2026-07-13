"""The typed adapter protocol every supported model implements.

Adapters wrap an already-loaded model and expose exactly what the lens
machinery needs: differentiable residual-stream access at one consistent,
explicitly declared location, the final normalization + unembedding, and
position metadata (modality classification for VLMs).
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, runtime_checkable

import torch
from torch import nn

from openjspace.types import (
    ForwardResult,
    ModelInputs,
    PositionMetadata,
    ResidualLocation,
    TokenizedInput,
)


@runtime_checkable
class LensModelAdapter(Protocol):
    """What OpenJSpace needs from a model.

    Attributes:
        model_id: Hugging Face id (or synthetic id for test models).
        model_revision: Checkout revision when known, else ``None``.
        architecture: Model class name (e.g. ``Qwen2ForCausalLM``).
        n_layers: Number of residual blocks in the *language* decoder.
        hidden_size: Residual-stream width of the language decoder.
        vocab_size: Unembedding output size.
        tokenizer_id: Tokenizer identifier for artifact validation.
        residual_location: Which tensor is captured as the residual stream.
            One consistent location across all layers; recorded in lens
            metadata so incompatible lenses cannot be silently applied.
        device: Device the language decoder inputs live on.
    """

    model_id: str
    model_revision: str | None
    architecture: str
    n_layers: int
    hidden_size: int
    vocab_size: int
    tokenizer_id: str
    residual_location: ResidualLocation

    @property
    def device(self) -> str:
        """Device the language decoder inputs live on."""
        ...

    def tokenize_text(self, text: str, *, max_length: int = 512) -> TokenizedInput:
        """Tokenize raw text (no chat template) to ``[1, seq_len]`` ids."""
        ...

    def prepare_inputs(
        self,
        prompt: str,
        *,
        images: Sequence[object] | None = None,
        max_length: int = 512,
        use_chat_template: bool = False,
    ) -> ModelInputs:
        """Build the exact model inputs plus per-position metadata.

        VLM adapters must use the official processor/chat template and
        preserve the exact multimodal sequence sent to the model.
        """
        ...

    def get_residual_modules(self) -> Sequence[nn.Module]:
        """The residual blocks, indexable by layer, that hooks attach to."""
        ...

    def get_final_norm(self) -> nn.Module:
        """The model's final normalization module (pre-unembedding)."""
        ...

    def get_unembedding_weight(self) -> torch.Tensor:
        """The unembedding matrix ``W_U`` of shape ``[vocab, hidden]``."""
        ...

    def decode_token_ids(self, token_ids: Sequence[int]) -> list[str]:
        """Decode each id to its raw string form (one string per id)."""
        ...

    def classify_positions(self, inputs: ModelInputs) -> list[PositionMetadata]:
        """Modality metadata for every sequence position of ``inputs``."""
        ...

    def forward_with_activations(
        self,
        inputs: ModelInputs,
        layers: Sequence[int],
        *,
        grad_from: int | None = None,
        replicate_batch: int = 1,
    ) -> ForwardResult:
        """Run one forward pass and record residuals at ``layers``.

        Args:
            inputs: Exact inputs from :meth:`prepare_inputs`.
            layers: Layer indices to record.
            grad_from: When set, run with gradients enabled and root the
                autograd graph at this layer's captured residual (used by
                fitting). When ``None``, run in inference mode.
            replicate_batch: Replicate the input along the batch axis (the
                fitting estimator computes ``replicate_batch`` Jacobian rows
                per backward pass). Requires deterministic behaviour across
                batch elements (eval mode, dropout off).
        """
        ...

    def unembed(self, residual: torch.Tensor) -> torch.Tensor:
        """Map residuals ``[..., hidden]`` to logits ``[..., vocab]``.

        Applies the model's own final normalization then the unembedding
        (plus final logit softcapping when the architecture uses it).
        """
        ...
