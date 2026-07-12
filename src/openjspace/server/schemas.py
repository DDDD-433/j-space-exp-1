"""Pydantic request/response schemas for the local API."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

SAFE_NAME_PATTERN = r"^[A-Za-z0-9][A-Za-z0-9._-]*$"


class LoadModelRequest(BaseModel):
    model_id: str = Field(min_length=1, max_length=256)
    device: str = "auto"
    dtype: str = "auto"
    revision: str | None = None


class ModelInfo(BaseModel):
    model_id: str
    architecture: str
    kind: Literal["text", "vlm"]
    status: str
    n_layers: int
    hidden_size: int
    vocab_size: int
    residual_location: str
    tokenizer_id: str
    device: str
    dtype: str


class FitRequest(BaseModel):
    model_id: str = Field(min_length=1, max_length=256)
    dataset: str = "wikitext"
    artifact_name: str = Field(pattern=SAFE_NAME_PATTERN, max_length=128)
    layers: list[int] | None = None
    target_layer: int | None = None
    max_seq_len: int = Field(default=128, ge=8, le=4096)
    num_prompts: int = Field(default=100, ge=1, le=100_000)
    dim_batch: int = Field(default=8, ge=1, le=512)
    skip_first: int = Field(default=16, ge=0, le=1024)
    seed: int = 0
    device: str = "auto"
    dtype: str = "auto"
    checkpoint_every: int = Field(default=10, ge=1)


class MergeRequest(BaseModel):
    shard_names: list[str] = Field(min_length=1)
    output_name: str = Field(pattern=SAFE_NAME_PATTERN, max_length=128)


class InspectRequest(BaseModel):
    model_id: str = Field(min_length=1, max_length=256)
    lens_name: str = Field(min_length=1, max_length=256)
    prompt: str = Field(min_length=1, max_length=100_000)
    image_names: list[str] = Field(default_factory=list)
    layers: list[int] | None = None
    positions: str = "all"
    top_k: int = Field(default=10, ge=1, le=100)
    max_seq_len: int = Field(default=512, ge=8, le=8192)
    use_chat_template: bool = False
    tracked_token_ids: list[int] = Field(default_factory=list)
    tracked_concepts: list[str] = Field(default_factory=list)
    device: str = "auto"
    dtype: str = "auto"
    force: bool = False


class DecomposeRequest(BaseModel):
    model_id: str = Field(min_length=1, max_length=256)
    lens_name: str = Field(min_length=1, max_length=256)
    prompt: str = Field(min_length=1, max_length=100_000)
    image_names: list[str] = Field(default_factory=list)
    layer: int
    position: int
    k: int = Field(default=10, ge=1, le=128)
    use_chat_template: bool = False
    max_seq_len: int = Field(default=512, ge=8, le=8192)
    device: str = "auto"
    dtype: str = "auto"


class JobInfo(BaseModel):
    job_id: str
    kind: str
    status: Literal["queued", "running", "succeeded", "failed", "cancelled"]
    progress: int = 0
    total: int = 0
    logs: list[str] = Field(default_factory=list)
    error: str | None = None
    checkpoint_path: str | None = None
    result: dict | None = None


class ArtifactInfo(BaseModel):
    name: str
    path: str
    metadata: dict


class RunInfo(BaseModel):
    run_id: str
    created_at: str
    model_id: str
    prompt: str


class UploadInfo(BaseModel):
    name: str
    width: int
    height: int
