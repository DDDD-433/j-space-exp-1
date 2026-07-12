"""Qwen2/Qwen2.5 adapter.

Qwen2-family checkpoints follow the modern HF decoder layout exactly
(``model.layers`` / ``model.norm`` / ``model.embed_tokens`` / ``lm_head``), so
the adapter is the generic :class:`~openjspace.models.hf_decoder.HFDecoderAdapter`
with the layout pinned explicitly (no auto-detection surprises) and one
Qwen-specific note: Qwen tokenizers have no BOS token, so raw-text prompts are
fitted exactly as tokenized.

This is the primary development family; see the integration test in
``tests/test_integration_qwen.py``.
"""

from __future__ import annotations

from typing import Any

from torch import nn

from openjspace.models.hf_decoder import HFDecoderAdapter, Layout

QWEN_LAYOUT = Layout(path="model")


class QwenDecoderAdapter(HFDecoderAdapter):
    """:class:`HFDecoderAdapter` pinned to the Qwen2 decoder layout."""

    def __init__(
        self,
        hf_model: nn.Module,
        tokenizer: Any,
        *,
        model_id: str = "",
        model_revision: str | None = None,
    ) -> None:
        super().__init__(
            hf_model,
            tokenizer,
            model_id=model_id,
            model_revision=model_revision,
            layout=QWEN_LAYOUT,
            # Qwen has no BOS; forcing one would corrupt the sequence.
            force_bos=False,
        )
