"""Normalization handling for lens readouts.

The J-lens readout is ``z = W_U · Norm(J_l · h)`` where ``Norm`` is the
model's own final normalization (RMSNorm/LayerNorm). Two subtleties are
centralized here:

1. The norm and unembedding run in the model's weight dtype on the model's
   device; residuals transported in float32 are cast in.
2. For inner-product analyses (per-token probes, sparse decomposition) the
   nonlinear ``Norm`` is *omitted*: pre-softmax logits are determined,
   approximately and up to a data-dependent scale, by the inner products
   ``<v_t, h>`` with the linear J-lens vectors ``v_t = (W_U J_l)[t]``. This
   matches the paper's probe/decomposition usage and keeps the dictionary
   linear. The approximation is documented in METHODOLOGY.md.
"""

from __future__ import annotations

import torch

from openjspace.models.protocol import LensModelAdapter


def apply_final_normalization(adapter: LensModelAdapter, residual: torch.Tensor) -> torch.Tensor:
    """Apply the model's own final normalization in its native dtype/device."""
    norm = adapter.get_final_norm()
    weight = adapter.get_unembedding_weight()
    return norm(residual.to(weight.dtype).to(weight.device))


def readout_logits(
    adapter: LensModelAdapter,
    residual: torch.Tensor,
    *,
    transport: torch.Tensor | None = None,
) -> torch.Tensor:
    """Full lens readout ``W_U · Norm(J · h)`` (or ``W_U · Norm(h)`` when
    ``transport`` is ``None``, i.e. the logit-lens baseline)."""
    if transport is not None:
        residual = residual.float() @ transport.to(residual.device).T
    return adapter.unembed(residual)


def jlens_vectors(
    adapter: LensModelAdapter,
    transport: torch.Tensor,
    token_ids: torch.Tensor | None = None,
) -> torch.Tensor:
    """Linear J-lens vectors at a layer: rows of ``W_U · J_l``.

    Args:
        adapter: Model adapter (provides ``W_U``).
        transport: The layer's ``J_l`` matrix ``[hidden, hidden]`` (float32).
        token_ids: Optional subset of vocabulary ids; ``None`` returns all.

    Returns:
        ``[n_tokens, hidden]`` float32 tensor on the transport's device; row
        ``i`` is the residual-space direction associated with token ``i``
        (before the final normalization — see module docstring).
    """
    weight = adapter.get_unembedding_weight().float().to(transport.device)
    if token_ids is not None:
        weight = weight[token_ids.to(weight.device)]
    return weight @ transport
