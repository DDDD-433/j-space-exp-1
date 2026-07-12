# Portions derived from anthropics/jacobian-lens (jlens/fitting.py)
# Copyright 2026 Anthropic PBC
# Copyright 2026 The OpenJSpace Authors
# SPDX-License-Identifier: Apache-2.0
"""Fitting the Jacobian lens.

The lens reads out an intermediate residual ``h_l`` by linearly transporting
it into the final-layer basis with the average input-output Jacobian::

    J_l = E[ dh_{L,t'} / dh_{l,t} ]   over prompts x, source positions t,
                                      and target positions t' >= t

Estimator (:func:`jacobian_for_prompt`), following the official
implementation: for each output dimension, inject a one-hot cotangent at
*every valid target position at once* and backprop. By causality of the
decoder, the gradient at source position ``p`` is then
``sum_{p' >= p} dh_L[p'] / dh_l[p]`` — the sum over current-and-future target
positions; we take the mean over valid source positions ``p``, then the mean
over prompts. Accumulation is float32 on CPU.

Cost: one forward pass (prompt replicated ``dim_batch`` times along the batch
axis) and ``ceil(hidden / dim_batch)`` backward passes per prompt. Shard
across machines by fitting disjoint prompt slices and merging with
:meth:`openjspace.core.lens.JacobianLens.merge` (count-weighted).

Never compute finite-difference Jacobians here; those exist only in tiny unit
tests as a cross-check of this estimator.
"""

from __future__ import annotations

import gc
import json
import logging
import math
import os
import time
from collections.abc import Callable, Sequence
from pathlib import Path

import torch
from safetensors.torch import load_file as safetensors_load

from openjspace.config import empty_device_cache
from openjspace.core.lens import JacobianLens
from openjspace.core.serialization import (
    atomic_save_safetensors,
    atomic_write_json,
    new_metadata,
)
from openjspace.models.protocol import LensModelAdapter

logger = logging.getLogger(__name__)

#: Positions before this index are excluded from the Jacobian average; early
#: positions act as attention sinks and have atypical residual statistics
#: (upstream default, paper §Methods).
SKIP_FIRST_N_POSITIONS = 16


class FittingCancelled(RuntimeError):
    """Raised when a cancellation callback asked the fit loop to stop."""


def valid_position_mask(seq_len: int, *, skip_first: int = SKIP_FIRST_N_POSITIONS) -> torch.Tensor:
    """Boolean mask over sequence positions included in the Jacobian average.

    Early positions are dominated by attention-sink behaviour and the final
    position has no next-token target, so both are excluded.

    Args:
        seq_len: Length of the tokenized prompt.
        skip_first: Number of leading positions to exclude.

    Returns:
        Boolean tensor of shape ``[seq_len]``.

    Raises:
        ValueError: If ``skip_first`` is negative or the prompt is too short
            to leave any valid positions.
    """
    if skip_first < 0:
        raise ValueError(f"skip_first must be >= 0, got {skip_first}")
    mask = torch.zeros(seq_len, dtype=torch.bool)
    mask[skip_first : seq_len - 1] = True
    if mask.sum() == 0:
        raise ValueError(f"prompt too short: seq_len={seq_len}, need > {skip_first + 1} tokens")
    return mask


def check_layer_indices(
    source_layers: Sequence[int] | None, target_layer: int | None, n_layers: int
) -> tuple[list[int], int]:
    """Resolve None/negative layer indices, bounds-check, enforce source < target."""
    target = n_layers - 1 if target_layer is None else target_layer
    if target < 0:
        target += n_layers
    if not 0 <= target < n_layers:
        raise ValueError(f"target_layer={target_layer} out of range for {n_layers} layers")
    if source_layers is None:
        return list(range(target)), target
    sources = sorted({layer + n_layers if layer < 0 else layer for layer in source_layers})
    if not sources or sources[0] < 0 or sources[-1] >= n_layers:
        raise ValueError(
            f"source_layers {sorted(source_layers)} out of range for {n_layers} layers"
        )
    if sources[-1] >= target:
        raise ValueError(
            f"source_layers must all be < target_layer={target}; got max={sources[-1]}"
        )
    return sources, target


def jacobian_for_prompt(
    adapter: LensModelAdapter,
    prompt: str,
    source_layers: Sequence[int],
    *,
    target_layer: int | None = None,
    dim_batch: int = 8,
    max_seq_len: int = 128,
    skip_first: int = SKIP_FIRST_N_POSITIONS,
) -> tuple[dict[int, torch.Tensor], int, int]:
    """Compute the per-layer Jacobian estimator ``J_l`` for one prompt.

    Runs one forward pass on the prompt replicated ``dim_batch`` times along
    the batch axis, retains the graph, then runs ``ceil(hidden / dim_batch)``
    backward passes against it. Each backward computes ``dim_batch`` rows of
    ``J_l`` at once: batch element ``b`` carries a one-hot cotangent at output
    dimension ``dim_start + b``, set at every valid target position.

    Args:
        adapter: Model adapter to compute Jacobians for.
        prompt: Input text (raw; fitting uses no chat template).
        source_layers: Layer indices ``l`` to compute ``J_l`` at.
        target_layer: Layer to take gradients of. Defaults to the final layer;
            negative indices count from the end. Targeting the penultimate
            layer can sometimes give a better-conditioned ``J_l``.
        dim_batch: Output dimensions computed per backward pass. Higher uses
            more memory (the prompt is replicated this many times); total
            backward FLOPs are unchanged.
        max_seq_len: Truncate the prompt to this many tokens.
        skip_first: Leading positions to exclude; see :func:`valid_position_mask`.

    Returns:
        ``(jacobians, seq_len, n_valid_positions)``. ``jacobians`` maps each
        source layer to a ``[hidden, hidden]`` fp32 CPU tensor.

    Raises:
        ValueError: If layer indices are invalid or the prompt is too short.
        RuntimeError: If gradients come back non-finite (NaN/Inf).
    """
    n_layers, hidden = adapter.n_layers, adapter.hidden_size
    sources, target = check_layer_indices(source_layers, target_layer, n_layers)

    inputs = adapter.prepare_inputs(prompt, max_length=max_seq_len)
    seq_len = inputs.seq_len
    position_mask = valid_position_mask(seq_len, skip_first=skip_first)
    n_valid_positions = int(position_mask.sum())

    jacobians = {layer: torch.zeros(hidden, hidden, dtype=torch.float32) for layer in sources}
    n_passes = math.ceil(hidden / dim_batch)

    with torch.enable_grad():
        result = adapter.forward_with_activations(
            inputs,
            layers=[*sources, target],
            grad_from=min(sources),
            replicate_batch=dim_batch,
        )
        target_activation = result.activations[target]  # [dim_batch, seq_len, hidden]
        source_activations = [result.activations[layer] for layer in sources]

        valid_positions = position_mask.nonzero(as_tuple=True)[0].to(target_activation.device)
        batch_indices = torch.arange(dim_batch, device=target_activation.device)
        cotangent = torch.zeros_like(target_activation)

        for pass_idx, dim_start in enumerate(range(0, hidden, dim_batch)):
            n_dims = min(dim_batch, hidden - dim_start)
            # One-hot cotangent at dim (dim_start + b) for batch element b, at
            # every valid target position. Yields rows dim_start..+n of J_l.
            cotangent.zero_()
            cotangent[
                batch_indices[:n_dims, None],
                valid_positions[None, :],
                dim_start + batch_indices[:n_dims, None],
            ] = 1.0
            grads = torch.autograd.grad(
                outputs=target_activation,
                inputs=source_activations,
                grad_outputs=cotangent,
                retain_graph=(pass_idx < n_passes - 1),
            )
            for layer, grad in zip(sources, grads, strict=True):
                positions_on_device = valid_positions.to(grad.device, non_blocking=True)
                rows = grad[:n_dims, positions_on_device, :].float().mean(dim=1)
                if not torch.isfinite(rows).all():
                    raise RuntimeError(
                        f"non-finite gradients at layer {layer}, dims "
                        f"{dim_start}..{dim_start + n_dims}; check model dtype/device"
                    )
                jacobians[layer][dim_start : dim_start + n_dims, :] = rows.cpu()
            del grads

    del result, target_activation, source_activations, cotangent
    return jacobians, seq_len, n_valid_positions


class FitCheckpoint:
    """Resumable running state of a fit: layer sums + counters.

    Stored as ``<path>.safetensors`` (running float32 sums) plus
    ``<path>.json`` (counters and fitting hyperparameters). Writes are atomic.
    """

    def __init__(self, path: str | Path) -> None:
        base = Path(path)
        self.tensor_path = base.with_suffix(".safetensors")
        self.meta_path = base.with_suffix(".json")

    def exists(self) -> bool:
        return self.tensor_path.is_file() and self.meta_path.is_file()

    def save(
        self,
        jacobian_sum: dict[int, torch.Tensor],
        *,
        n_done: int,
        next_idx: int,
        n_valid_total: int,
        source_layers: list[int],
        target_layer: int,
        skip_first: int,
    ) -> None:
        atomic_save_safetensors(
            self.tensor_path, {f"J_{layer}": J for layer, J in jacobian_sum.items()}
        )
        atomic_write_json(
            self.meta_path,
            {
                "n_done": n_done,
                "next_idx": next_idx,
                "n_valid_total": n_valid_total,
                "source_layers": source_layers,
                "target_layer": target_layer,
                "skip_first": skip_first,
            },
        )

    def load(
        self, *, source_layers: list[int], target_layer: int, skip_first: int
    ) -> tuple[dict[int, torch.Tensor], int, int, int]:
        """Load and validate the checkpoint against current fit settings.

        Raises:
            ValueError: If the checkpoint was fitted with different settings.
        """
        meta = json.loads(self.meta_path.read_text())
        for key, expected in (
            ("source_layers", source_layers),
            ("target_layer", target_layer),
            ("skip_first", skip_first),
        ):
            if meta.get(key) != expected:
                raise ValueError(
                    f"checkpoint at {self.meta_path} was fitted with {key}="
                    f"{meta.get(key)!r}, not {expected!r}; delete it or change settings"
                )
        raw = safetensors_load(str(self.tensor_path))
        jacobian_sum = {int(key[2:]): tensor.float() for key, tensor in raw.items()}
        return jacobian_sum, meta["n_done"], meta["next_idx"], meta.get("n_valid_total", 0)


ProgressCallback = Callable[[int, int, str], None]
"""``(prompts_done, prompts_total, message)`` — called after every prompt."""


def fit(
    adapter: LensModelAdapter,
    prompts: Sequence[str],
    *,
    source_layers: Sequence[int] | None = None,
    target_layer: int | None = None,
    dim_batch: int = 8,
    max_seq_len: int = 128,
    skip_first: int = SKIP_FIRST_N_POSITIONS,
    checkpoint_path: str | Path | None = None,
    checkpoint_every: int | None = 10,
    resume: bool = True,
    fitting_dataset: str = "",
    model_dtype: str = "",
    progress: ProgressCallback | None = None,
    should_cancel: Callable[[], bool] | None = None,
) -> JacobianLens:
    """Fit ``J_l`` over prompts and return a :class:`JacobianLens`.

    Per-prompt Jacobians from :func:`jacobian_for_prompt` are accumulated as a
    float32 running sum. If ``checkpoint_path`` is set, the running sum is
    written atomically every ``checkpoint_every`` prompts and resumed from on
    restart (a prompt skipped for being too short is not re-processed).

    Args:
        adapter: The model adapter to fit on.
        prompts: Text prompts to average over. The paper's lenses use 1000
            sequences of 128 tokens from a pretraining-like corpus; quality
            saturates quickly and ~100 prompts is usable.
        source_layers: Layers to fit at. Defaults to every layer below
            ``target_layer``; negative indices count from the end.
        target_layer: See :func:`jacobian_for_prompt`.
        dim_batch: See :func:`jacobian_for_prompt`.
        max_seq_len: Truncate each prompt to this many tokens.
        skip_first: See :func:`valid_position_mask`.
        checkpoint_path: If set, write a resumable checkpoint here (base path;
            ``.safetensors``/``.json`` suffixes are added).
        checkpoint_every: Write the checkpoint every N prompts. ``None`` saves
            only at the end (checkpoints are ``len(source_layers) * hidden**2 *
            4`` bytes, so raise this for large models).
        resume: If ``True`` and the checkpoint exists, resume from it.
        fitting_dataset: Free-form dataset description stored in metadata.
        model_dtype: Model weight dtype string stored in metadata.
        progress: Optional callback invoked after each prompt.
        should_cancel: Optional callback polled between prompts; returning
            ``True`` checkpoints and raises :class:`FittingCancelled`.

    Returns:
        The fitted :class:`JacobianLens`.

    Raises:
        ValueError: If no prompt was long enough to fit on.
        FittingCancelled: If ``should_cancel`` requested a stop.
    """
    n_layers, hidden = adapter.n_layers, adapter.hidden_size
    sources, target = check_layer_indices(source_layers, target_layer, n_layers)

    logger.info(
        "fit: model=%s n_layers=%d hidden=%d, fitting %d source layers (target=L%d) on %d prompts",
        adapter.model_id,
        n_layers,
        hidden,
        len(sources),
        target,
        len(prompts),
    )

    checkpoint = FitCheckpoint(checkpoint_path) if checkpoint_path is not None else None
    jacobian_sum: dict[int, torch.Tensor]
    if resume and checkpoint is not None and checkpoint.exists():
        jacobian_sum, n_done, next_idx, n_valid_total = checkpoint.load(
            source_layers=sources, target_layer=target, skip_first=skip_first
        )
        logger.info("  resuming from checkpoint: %d/%d prompts processed", next_idx, len(prompts))
    else:
        jacobian_sum = {
            layer: torch.zeros(hidden, hidden, dtype=torch.float32) for layer in sources
        }
        n_done, next_idx, n_valid_total = 0, 0, 0

    def write_checkpoint() -> None:
        if checkpoint is not None:
            checkpoint.save(
                jacobian_sum,
                n_done=n_done,
                next_idx=next_idx,
                n_valid_total=n_valid_total,
                source_layers=sources,
                target_layer=target,
                skip_first=skip_first,
            )

    sqrt_d = math.sqrt(hidden)
    for prompt_idx, prompt in enumerate(prompts):
        if prompt_idx < next_idx:
            continue
        if should_cancel is not None and should_cancel():
            write_checkpoint()
            raise FittingCancelled(f"fit cancelled after {n_done} prompts")
        start_time = time.perf_counter()
        try:
            per_prompt_J, seq_len, n_valid = jacobian_for_prompt(
                adapter,
                prompt,
                sources,
                target_layer=target,
                dim_batch=dim_batch,
                max_seq_len=max_seq_len,
                skip_first=skip_first,
            )
        except ValueError as exc:
            logger.warning("  skipping prompt %d: %s", prompt_idx, exc)
            next_idx = prompt_idx + 1
            continue

        # Per-prompt diagnostics: the prompt's own Jacobian norm flags
        # heavy-tailed outliers; the relative shift of the running mean tracks
        # convergence (falls ~1/n once settled).
        prompt_norm = max(per_prompt_J[layer].norm().item() for layer in sources) / sqrt_d
        if n_done > 0:
            mean_rel_change = max(
                (
                    (per_prompt_J[layer] - jacobian_sum[layer] / n_done).norm()
                    / ((n_done + 1) * (jacobian_sum[layer] / n_done).norm())
                ).item()
                for layer in sources
            )
        else:
            mean_rel_change = float("nan")

        for layer in sources:
            jacobian_sum[layer] += per_prompt_J[layer]
        n_done += 1
        n_valid_total += n_valid
        next_idx = prompt_idx + 1

        message = (
            f"prompt {prompt_idx + 1}/{len(prompts)}  seq_len={seq_len} n_valid={n_valid}  "
            f"{time.perf_counter() - start_time:.0f}s  "
            f"max||J||/sqrt(d)={prompt_norm:.3f}  max_d_mean={mean_rel_change:.2e}"
        )
        logger.info("  %s", message)
        if progress is not None:
            progress(next_idx, len(prompts), message)
        if checkpoint_every is not None and next_idx % checkpoint_every == 0:
            write_checkpoint()
        del per_prompt_J
        gc.collect()
        empty_device_cache(adapter.device)

    write_checkpoint()
    if n_done == 0:
        raise ValueError("no prompts were long enough to fit on")
    jacobian_mean = {layer: jacobian_sum[layer] / n_done for layer in sources}
    logger.info("fit: done, %d prompts", n_done)

    metadata = new_metadata(
        model_id=adapter.model_id,
        model_revision=adapter.model_revision,
        model_architecture=adapter.architecture,
        tokenizer_id=adapter.tokenizer_id,
        hidden_size=hidden,
        vocab_size=adapter.vocab_size,
        n_layers=n_layers,
        source_layers=sources,
        target_layer=target,
        residual_location=adapter.residual_location,
        sequence_length=max_seq_len,
        skip_first_positions=skip_first,
        number_of_prompts=n_done,
        number_of_valid_positions=n_valid_total,
        dtype_used_for_model=model_dtype,
        fitting_dataset=fitting_dataset,
    )
    return JacobianLens(jacobian_mean, metadata)


def cleanup_checkpoint(checkpoint_path: str | Path) -> None:
    """Remove checkpoint files after a successful fit."""
    checkpoint = FitCheckpoint(checkpoint_path)
    for path in (checkpoint.tensor_path, checkpoint.meta_path):
        if path.is_file():
            os.remove(path)
