"""Local-first configuration: artifact directories and device selection."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import torch

ENV_HOME = "OPENJSPACE_HOME"


def openjspace_home() -> Path:
    """Root directory for artifacts and runs (default ``~/.openjspace``)."""
    return Path(os.environ.get(ENV_HOME, str(Path.home() / ".openjspace")))


def artifacts_dir() -> Path:
    return openjspace_home() / "artifacts"


def runs_dir() -> Path:
    return openjspace_home() / "runs"


def uploads_dir() -> Path:
    return openjspace_home() / "uploads"


@dataclass(frozen=True)
class DeviceInfo:
    device: str
    supports_bf16: bool
    supports_fp16: bool


def resolve_device(requested: str = "auto") -> str:
    """Resolve ``auto``/``cuda``/``mps``/``cpu`` to an available device string.

    Raises:
        ValueError: If an explicitly requested device is not available.
    """
    if requested == "auto":
        if torch.cuda.is_available():
            return "cuda"
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    if requested.startswith("cuda") and not torch.cuda.is_available():
        raise ValueError("CUDA requested but torch.cuda.is_available() is False")
    if requested == "mps" and not (
        getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()
    ):
        raise ValueError("MPS requested but not available on this system")
    if requested not in ("cpu", "mps") and not requested.startswith("cuda"):
        raise ValueError(f"unknown device {requested!r}; expected auto/cuda/mps/cpu")
    return requested


def device_info(device: str) -> DeviceInfo:
    if device.startswith("cuda"):
        return DeviceInfo(
            device=device,
            supports_bf16=torch.cuda.is_bf16_supported(),
            supports_fp16=True,
        )
    if device == "mps":
        return DeviceInfo(device=device, supports_bf16=False, supports_fp16=True)
    return DeviceInfo(device=device, supports_bf16=True, supports_fp16=False)


def resolve_dtype(requested: str, device: str) -> torch.dtype:
    """Resolve a dtype string, defaulting to the best precision for ``device``.

    ``auto`` uses bf16 on CUDA when supported, fp16 on MPS, fp32 on CPU.
    """
    table = {
        "float32": torch.float32,
        "fp32": torch.float32,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float16": torch.float16,
        "fp16": torch.float16,
    }
    if requested != "auto":
        try:
            return table[requested]
        except KeyError:
            raise ValueError(
                f"unknown dtype {requested!r}; expected auto/float32/bfloat16/float16"
            ) from None
    info = device_info(device)
    if device.startswith("cuda"):
        return torch.bfloat16 if info.supports_bf16 else torch.float16
    if device == "mps":
        return torch.float16
    return torch.float32


def empty_device_cache(device: str) -> None:
    """Release cached device memory after dropping tensor references."""
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif device == "mps" and hasattr(torch, "mps") and hasattr(torch.mps, "empty_cache"):
        torch.mps.empty_cache()
