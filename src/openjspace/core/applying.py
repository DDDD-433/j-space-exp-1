"""Applying a fitted Jacobian lens: the structured inspection pipeline.

Runs the model once while recording residual-stream activations, then reads
out each requested (layer, position) cell through three methods:

- **J-lens**: ``unembed(J_l · h)`` — transported into the target-layer basis.
- **Logit lens**: ``unembed(h)`` — the ``J = I`` baseline.
- **Model output**: the final layer's own readout (shown as the last row).

Outputs a :class:`openjspace.report.schema.RunResult` — top-K concepts per
cell, comparison metrics, activation norms, and rank grids for tracked
concepts. Full-vocabulary logits are not stored unless explicitly requested.
"""

from __future__ import annotations

import uuid
from collections.abc import Sequence
from datetime import UTC, datetime

import torch

from openjspace.analysis.comparisons import compare_cell
from openjspace.analysis.rankings import clean_token_text, normalized_scores, ranks_of
from openjspace.core.lens import JacobianLens
from openjspace.models.protocol import LensModelAdapter
from openjspace.report.schema import (
    CellRecord,
    ConceptEntry,
    PositionInfo,
    RunMetadata,
    RunResult,
    TrackedConcept,
)
from openjspace.types import ModelInputs


def _top_entries(adapter: LensModelAdapter, logits: torch.Tensor, top_k: int) -> list[ConceptEntry]:
    """Top-K concept entries for one cell's ``[vocab]`` logits."""
    values, indices = logits.topk(top_k)
    norm = normalized_scores(logits.unsqueeze(0), values.unsqueeze(0))[0]
    texts = adapter.decode_token_ids(indices.tolist())
    return [
        ConceptEntry(
            token_id=int(token_id),
            token_text=text,
            token_display=clean_token_text(text),
            score=float(score),
            normalized_score=float(normalized),
        )
        for token_id, text, score, normalized in zip(
            indices.tolist(), texts, values.tolist(), norm.tolist(), strict=True
        )
    ]


def select_positions(positions_spec: str | Sequence[int], seq_len: int) -> list[int]:
    """Resolve a positions spec (``"all"``, ``"last:N"``, or explicit indices,
    negative allowed) to sorted valid indices.

    Raises:
        ValueError: On out-of-range indices or an unknown spec string.
    """
    if isinstance(positions_spec, str):
        if positions_spec == "all":
            return list(range(seq_len))
        if positions_spec.startswith("last:"):
            n = int(positions_spec.split(":", 1)[1])
            return list(range(max(0, seq_len - n), seq_len))
        try:
            indices = [int(part) for part in positions_spec.split(",") if part.strip()]
        except ValueError:
            raise ValueError(
                f"positions spec {positions_spec!r} not understood; use 'all', "
                f"'last:N', or comma-separated indices"
            ) from None
    else:
        indices = list(positions_spec)
    resolved = sorted({i + seq_len if i < 0 else i for i in indices})
    bad = [i for i in resolved if not 0 <= i < seq_len]
    if bad:
        raise ValueError(f"positions {bad} out of range for sequence length {seq_len}")
    return resolved


@torch.no_grad()
def inspect_prompt(
    adapter: LensModelAdapter,
    lens: JacobianLens,
    prompt: str,
    *,
    images: Sequence[object] | None = None,
    layers: Sequence[int] | None = None,
    positions: str | Sequence[int] = "all",
    top_k: int = 10,
    max_seq_len: int = 512,
    use_chat_template: bool = False,
    tracked_token_ids: Sequence[int] = (),
    force: bool = False,
    lens_path: str = "",
    device: str = "",
    dtype: str = "",
    model_kind: str = "text",
    prepared_inputs: ModelInputs | None = None,
) -> RunResult:
    """Run one prompt and produce the full structured readout.

    Args:
        adapter: The model adapter.
        lens: A fitted, compatible :class:`JacobianLens`.
        prompt: Input text.
        images: Optional images for VLM adapters.
        layers: Lens layers to read out (defaults to all fitted layers). The
            model's final layer is always appended as the model-output row.
        positions: ``"all"``, ``"last:N"``, comma string, or index sequence.
        top_k: Concepts kept per cell.
        max_seq_len: Truncation length for the prompt.
        use_chat_template: Format the prompt with the tokenizer's chat template.
        tracked_token_ids: Concepts whose full rank/score grids are recorded.
        force: Apply the lens even when compatibility validation fails
            (mismatches become warnings in the run metadata).
        prepared_inputs: Pre-built inputs (used by VLM callers that already ran
            the processor); overrides ``prompt``/``images`` tokenization.

    Returns:
        A :class:`RunResult`.
    """
    warnings = [f"lens/model mismatch: {w}" for w in lens.validate_against(adapter, force=force)]

    if layers is None:
        layers = lens.source_layers
    unknown = sorted(set(layers) - set(lens.source_layers))
    if unknown:
        raise ValueError(
            f"layers {unknown} not fitted by this lens; fitted layers are {lens.source_layers}"
        )
    final_layer = adapter.n_layers - 1
    lens_layers = sorted(set(layers))
    record_layers = sorted({*lens_layers, final_layer})

    if prepared_inputs is not None:
        inputs = prepared_inputs
    else:
        inputs = adapter.prepare_inputs(
            prompt, images=images, max_length=max_seq_len, use_chat_template=use_chat_template
        )
    position_metadata = adapter.classify_positions(inputs)
    seq_len = inputs.seq_len
    selected = select_positions(positions, seq_len)

    result = adapter.forward_with_activations(inputs, layers=record_layers)
    activations = {
        layer: tensor[0].detach().clone() for layer, tensor in result.activations.items()
    }
    del result

    position_tensor = torch.tensor(selected, dtype=torch.long)
    tracked_ids = list(dict.fromkeys(int(t) for t in tracked_token_ids))
    tracked_texts = adapter.decode_token_ids(tracked_ids) if tracked_ids else []

    cells: list[CellRecord] = []
    n_rows = len(lens_layers) + (0 if final_layer in lens_layers else 1)
    tracked_ranks = {tid: [[-1] * len(selected) for _ in range(n_rows)] for tid in tracked_ids}
    tracked_scores = {tid: [[0.0] * len(selected) for _ in range(n_rows)] for tid in tracked_ids}
    row_layers = lens_layers + ([final_layer] if final_layer not in lens_layers else [])

    for row_idx, layer in enumerate(row_layers):
        residual = activations[layer][position_tensor.to(activations[layer].device)].float()
        is_output_row = layer == final_layer
        if is_output_row:
            # Final layer: J = I; this row is the model's actual output.
            transported = residual
        else:
            transported = lens.transport(residual, layer)
        jlens_logits = adapter.unembed(transported).float().cpu()
        logit_lens_logits = (
            jlens_logits if is_output_row else adapter.unembed(residual).float().cpu()
        )

        if tracked_ids:
            ids_tensor = torch.tensor(tracked_ids, dtype=torch.long)
            rank_grid = ranks_of(jlens_logits, ids_tensor)
            for col, _pos in enumerate(selected):
                for t_idx, tid in enumerate(tracked_ids):
                    tracked_ranks[tid][row_idx][col] = int(rank_grid[col, t_idx])
                    tracked_scores[tid][row_idx][col] = float(jlens_logits[col, tid])

        residual_norm = residual.norm(dim=-1).cpu()
        transported_norm = transported.norm(dim=-1).cpu()
        for col, pos in enumerate(selected):
            cells.append(
                CellRecord(
                    layer=layer,
                    position=pos,
                    jlens_top=_top_entries(adapter, jlens_logits[col], top_k),
                    logit_lens_top=_top_entries(adapter, logit_lens_logits[col], top_k),
                    comparison=compare_cell(jlens_logits[col], logit_lens_logits[col], top_k=top_k),
                    activation_norm=float(residual_norm[col]),
                    transported_norm=float(transported_norm[col]),
                    is_model_output=is_output_row,
                )
            )
        del jlens_logits, logit_lens_logits, residual, transported

    tracked = [
        TrackedConcept(
            token_id=tid,
            token_text=text,
            token_display=clean_token_text(text),
            ranks=tracked_ranks[tid],
            scores=tracked_scores[tid],
        )
        for tid, text in zip(tracked_ids, tracked_texts, strict=True)
    ]

    if inputs.patch_mapping != "exact" and any(
        p.modality == "image_token" for p in position_metadata
    ):
        warnings.append(
            f"patch mapping is {inputs.patch_mapping!r}: image-token positions are "
            "shown as a token strip / merged groups, not exact pixel coordinates"
        )

    metadata = RunMetadata(
        run_id=uuid.uuid4().hex[:12],
        created_at=datetime.now(UTC).isoformat(timespec="seconds"),
        model_id=adapter.model_id,
        model_architecture=adapter.architecture,
        model_kind=model_kind,  # type: ignore[arg-type]
        device=device or adapter.device,
        dtype=dtype,
        lens_path=lens_path,
        lens_metadata=lens.metadata.model_dump(),
        prompt=prompt,
        used_chat_template=use_chat_template,
        layers=row_layers,
        top_k=top_k,
        n_positions=len(selected),
        patch_mapping=inputs.patch_mapping,
        warnings=warnings,
    )
    return RunResult(
        metadata=metadata,
        positions=[PositionInfo(**vars(p)) for p in position_metadata],
        cells=cells,
        tracked=tracked,
    )
