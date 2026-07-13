"""Versioned lens artifact format: ``lens.safetensors`` + ``metadata.json``.

Tensors are stored with safetensors; metadata is a validated Pydantic model.
All writes are atomic (temp file + ``os.replace``) so a crash never leaves a
half-written artifact. Loading validates every tensor for shape, dtype and
finiteness before returning.
"""

from __future__ import annotations

import json
import os
import platform
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch
from pydantic import BaseModel, Field
from safetensors.torch import load_file as safetensors_load
from safetensors.torch import save_file as safetensors_save

from openjspace.types import ResidualLocation

FORMAT_VERSION = 1
LENS_FILENAME = "lens.safetensors"
METADATA_FILENAME = "metadata.json"


def library_versions() -> dict[str, str]:
    import safetensors

    versions = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "safetensors": safetensors.__version__,
    }
    try:
        import transformers

        versions["transformers"] = transformers.__version__
    except ImportError:
        pass
    return versions


class LensMetadata(BaseModel):
    """Everything needed to validate a lens against a model before applying it."""

    format_version: int = FORMAT_VERSION
    method: str = "jacobian_lens"
    model_id: str
    model_revision: str | None = None
    model_architecture: str = ""
    tokenizer_id: str = ""
    tokenizer_revision: str | None = None
    hidden_size: int
    vocab_size: int
    n_layers: int
    source_layers: list[int]
    target_layer: int
    residual_location: ResidualLocation = "block_output"
    sequence_length: int = 128
    skip_first_positions: int = 16
    number_of_prompts: int = 0
    number_of_valid_positions: int = 0
    dtype_used_for_model: str = ""
    dtype_used_for_accumulation: str = "float32"
    fitting_dataset: str = ""
    created_at: str = ""
    library_versions: dict[str, str] = Field(default_factory=dict)
    notes: str = ""


class LensCompatibilityError(ValueError):
    """A lens artifact does not match the model it is being applied to."""


def atomic_write_bytes(path: Path, data: bytes) -> None:
    """Write ``data`` to ``path`` atomically (temp file + ``os.replace``)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    tmp.write_bytes(data)
    os.replace(tmp, path)


def atomic_write_json(path: Path, obj: Any) -> None:
    atomic_write_bytes(path, json.dumps(obj, indent=2, ensure_ascii=False).encode())


def atomic_save_safetensors(path: Path, tensors: dict[str, torch.Tensor]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    safetensors_save(tensors, str(tmp))
    os.replace(tmp, path)


def new_metadata(**kwargs: Any) -> LensMetadata:
    """Build metadata with ``created_at`` and library versions filled in."""
    kwargs.setdefault("created_at", datetime.now(UTC).isoformat(timespec="seconds"))
    kwargs.setdefault("library_versions", library_versions())
    return LensMetadata(**kwargs)


def save_lens(
    directory: str | Path,
    jacobians: dict[int, torch.Tensor],
    metadata: LensMetadata,
    *,
    storage_dtype: torch.dtype = torch.float16,
) -> Path:
    """Write a lens artifact directory.

    Jacobians are stored as ``storage_dtype`` (default fp16: halves file size;
    entries are O(1) so range is not a constraint and fp16's extra mantissa
    bits beat bf16 here).
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    for layer, J in jacobians.items():
        if not torch.isfinite(J).all():
            raise ValueError(f"refusing to save lens: J_{layer} contains NaN/Inf")
    tensors = {f"J_{layer}": J.to(storage_dtype).contiguous() for layer, J in jacobians.items()}
    atomic_save_safetensors(directory / LENS_FILENAME, tensors)
    atomic_write_json(directory / METADATA_FILENAME, metadata.model_dump())
    return directory


def load_lens(directory: str | Path) -> tuple[dict[int, torch.Tensor], LensMetadata]:
    """Load and validate a lens artifact directory.

    Returns:
        ``(jacobians, metadata)`` with jacobians as float32 CPU tensors.

    Raises:
        FileNotFoundError: If the directory or its files are missing.
        ValueError: If tensors or metadata are malformed (wrong shapes, keys,
            non-finite values, unsupported format version).
    """
    directory = Path(directory)
    meta_path = directory / METADATA_FILENAME
    lens_path = directory / LENS_FILENAME
    if not meta_path.is_file() or not lens_path.is_file():
        raise FileNotFoundError(
            f"{directory} is not a lens artifact (need {LENS_FILENAME} + {METADATA_FILENAME})"
        )
    metadata = LensMetadata.model_validate(json.loads(meta_path.read_text()))
    if metadata.format_version > FORMAT_VERSION:
        raise ValueError(
            f"lens format_version={metadata.format_version} is newer than "
            f"supported version {FORMAT_VERSION}; upgrade openjspace"
        )
    raw = safetensors_load(str(lens_path))
    jacobians: dict[int, torch.Tensor] = {}
    for key, tensor in raw.items():
        if not key.startswith("J_"):
            raise ValueError(f"unexpected tensor key {key!r} in {lens_path}")
        layer = int(key[2:])
        expected = (metadata.hidden_size, metadata.hidden_size)
        if tuple(tensor.shape) != expected:
            raise ValueError(f"J_{layer} has shape {tuple(tensor.shape)}, expected {expected}")
        tensor = tensor.float()
        if not torch.isfinite(tensor).all():
            raise ValueError(f"J_{layer} contains NaN/Inf values")
        jacobians[layer] = tensor
    if sorted(jacobians) != sorted(metadata.source_layers):
        raise ValueError(
            f"tensor layers {sorted(jacobians)} disagree with metadata "
            f"source_layers {sorted(metadata.source_layers)}"
        )
    return jacobians, metadata


def check_compatibility(
    metadata: LensMetadata,
    *,
    model_id: str,
    architecture: str,
    hidden_size: int,
    vocab_size: int,
    n_layers: int,
    tokenizer_id: str,
    residual_location: str,
    model_revision: str | None = None,
    force: bool = False,
) -> list[str]:
    """Validate a lens against a model before applying it.

    Returns the list of mismatch descriptions (empty when compatible).

    Raises:
        LensCompatibilityError: On any mismatch, unless ``force`` is set (the
            caller should then surface the returned warnings prominently).
    """
    problems: list[str] = []
    hard = [
        ("hidden_size", metadata.hidden_size, hidden_size),
        ("vocab_size", metadata.vocab_size, vocab_size),
        ("n_layers", metadata.n_layers, n_layers),
        ("residual_location", metadata.residual_location, residual_location),
    ]
    soft = [
        ("model_id", metadata.model_id, model_id),
        ("model_architecture", metadata.model_architecture, architecture),
        ("tokenizer_id", metadata.tokenizer_id, tokenizer_id),
    ]
    for name, lens_value, model_value in hard + soft:
        if lens_value and model_value and lens_value != model_value:
            problems.append(f"{name}: lens has {lens_value!r}, model has {model_value!r}")
    if metadata.model_revision and model_revision and metadata.model_revision != model_revision:
        problems.append(
            f"model_revision: lens has {metadata.model_revision!r}, model has {model_revision!r}"
        )
    if problems and not force:
        raise LensCompatibilityError(
            "lens artifact is not compatible with this model:\n  - "
            + "\n  - ".join(problems)
            + "\nPass force=True (--force) to apply anyway at your own risk."
        )
    return problems
