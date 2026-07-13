"""Run a sparse J-space decomposition for one (layer, position) cell."""

from __future__ import annotations

from collections.abc import Sequence

import torch

from openjspace.analysis.rankings import clean_token_text
from openjspace.core.decomposition import nonnegative_omp
from openjspace.core.lens import JacobianLens
from openjspace.core.normalization import jlens_vectors
from openjspace.models.protocol import LensModelAdapter
from openjspace.report.schema import DecompositionEntry, DecompositionRecord
from openjspace.types import ModelInputs


@torch.no_grad()
def decompose_cell(
    adapter: LensModelAdapter,
    lens: JacobianLens,
    prompt: str,
    *,
    layer: int,
    position: int,
    k: int = 10,
    images: Sequence[object] | None = None,
    use_chat_template: bool = False,
    max_seq_len: int = 512,
    prepared_inputs: ModelInputs | None = None,
    candidate_token_ids: torch.Tensor | None = None,
) -> DecompositionRecord:
    """Decompose one residual activation against the layer's J-lens vectors.

    Solves ``h ~= sum_i alpha_i v_i`` with ``alpha_i >= 0`` and at most ``k``
    atoms, where ``v_i`` are rows of ``W_U J_l`` (linear J-lens vectors; the
    final normalization is intentionally omitted — see
    :mod:`openjspace.core.normalization`).

    Args:
        candidate_token_ids: Optional vocabulary subset to use as the
            dictionary (full vocabulary by default; restrict for speed on very
            large vocabularies).

    Raises:
        ValueError: If ``layer`` is not fitted or ``position`` out of range.
    """
    if layer not in lens.source_layers:
        raise ValueError(f"layer {layer} not fitted; fitted layers are {lens.source_layers}")
    inputs = prepared_inputs or adapter.prepare_inputs(
        prompt, images=images, max_length=max_seq_len, use_chat_template=use_chat_template
    )
    seq_len = inputs.seq_len
    resolved = position + seq_len if position < 0 else position
    if not 0 <= resolved < seq_len:
        raise ValueError(f"position {position} out of range for sequence length {seq_len}")

    result = adapter.forward_with_activations(inputs, layers=[layer])
    activation = result.activations[layer][0, resolved].detach().float().cpu()

    transport = lens.jacobians[layer]  # [hidden, hidden] float32 CPU
    vectors = jlens_vectors(adapter, transport, token_ids=candidate_token_ids).cpu()
    dictionary = vectors.T  # [hidden, n_atoms]

    decomposition = nonnegative_omp(dictionary, activation, k=k)
    if candidate_token_ids is not None:
        token_ids = [int(candidate_token_ids[i]) for i in decomposition.indices]
    else:
        token_ids = decomposition.indices
    texts = adapter.decode_token_ids(token_ids)
    entries = [
        DecompositionEntry(
            token_id=tid,
            token_text=text,
            token_display=clean_token_text(text),
            coefficient=coefficient,
        )
        for tid, text, coefficient in zip(token_ids, texts, decomposition.coefficients, strict=True)
    ]
    return DecompositionRecord(
        layer=layer,
        position=resolved,
        entries=entries,
        k_requested=k,
        reconstruction_error=decomposition.reconstruction_error,
        residual_norm=decomposition.residual_norm,
        explained_norm_fraction=decomposition.explained_norm_fraction,
        n_iterations=decomposition.n_iterations,
    )
