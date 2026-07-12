"""Versioned, JSON-serializable schema for one inspection run.

This schema is the single source of truth consumed by the CLI (JSON export),
the web UI, and the self-contained HTML report. Vocabulary-sized tensors are
never stored here by default: cells carry top-K entries plus ranks of tracked
concepts.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

RUN_SCHEMA_VERSION = 1

SCIENTIFIC_DISCLAIMER = (
    "Jacobian-lens readouts are approximate concept readouts: corpus-averaged, "
    "linearized transports of residual-stream activations decoded through the "
    "model's own unembedding. They surface verbalizable activation directions "
    "— not literal 'thoughts' — and scores are rankings, not calibrated "
    "probabilities. Causal claims require interventions, not visualization."
)


class ConceptEntry(BaseModel):
    """One vocabulary concept in a top-K readout."""

    token_id: int
    token_text: str  # raw tokenizer output, never destroyed
    token_display: str  # cleaned display form
    score: float  # raw lens logit
    normalized_score: float  # [0,1] display normalization, not a probability


class PositionInfo(BaseModel):
    index: int
    modality: Literal["text", "image_token", "image_boundary", "special", "unknown"]
    token_id: int | None = None
    token_text: str | None = None
    image_index: int | None = None
    patch_index: int | None = None
    patch_row: int | None = None
    patch_col: int | None = None


class CellRecord(BaseModel):
    """Readouts at one (layer, position) cell."""

    layer: int
    position: int
    jlens_top: list[ConceptEntry]
    logit_lens_top: list[ConceptEntry]
    comparison: dict[str, float] = Field(default_factory=dict)
    activation_norm: float = 0.0
    transported_norm: float = 0.0
    is_model_output: bool = False
    """True on the final-layer row, where the readout (J = I) is the model's
    actual output distribution rather than a lens estimate."""


class TrackedConcept(BaseModel):
    """Full layer x position rank grid for one pinned/tracked token."""

    token_id: int
    token_text: str
    token_display: str
    ranks: list[list[int]]  # [n_layers][n_positions], -1 = not computed
    scores: list[list[float]] = Field(default_factory=list)


class DecompositionEntry(BaseModel):
    token_id: int
    token_text: str
    token_display: str
    coefficient: float  # always >= 0


class DecompositionRecord(BaseModel):
    """Sparse non-negative J-space decomposition of one cell's activation."""

    layer: int
    position: int
    entries: list[DecompositionEntry]
    k_requested: int
    reconstruction_error: float  # ||h - h_hat|| (transported basis)
    residual_norm: float
    explained_norm_fraction: float  # 1 - (residual/||h||)^2, clamped to [0,1]
    n_iterations: int
    warning: str = (
        "This decomposition is non-unique: the J-lens dictionary is correlated "
        "and overcomplete, so other sparse combinations may fit comparably well."
    )


class ImageInfo(BaseModel):
    image_index: int
    filename: str = ""
    width: int = 0
    height: int = 0
    grid_rows: int | None = None
    grid_cols: int | None = None
    data_uri: str = ""  # optional inline preview for reports


class RunMetadata(BaseModel):
    run_id: str
    created_at: str
    model_id: str
    model_architecture: str = ""
    model_kind: Literal["text", "vlm"] = "text"
    device: str = ""
    dtype: str = ""
    lens_path: str = ""
    lens_metadata: dict = Field(default_factory=dict)
    prompt: str
    used_chat_template: bool = False
    layers: list[int] = Field(default_factory=list)
    top_k: int = 10
    n_positions: int = 0
    patch_mapping: Literal["exact", "approximate", "unavailable"] = "unavailable"
    disclaimer: str = SCIENTIFIC_DISCLAIMER
    warnings: list[str] = Field(default_factory=list)


class RunResult(BaseModel):
    """Everything produced by one inspection run."""

    schema_version: int = RUN_SCHEMA_VERSION
    metadata: RunMetadata
    positions: list[PositionInfo]
    cells: list[CellRecord]
    tracked: list[TrackedConcept] = Field(default_factory=list)
    decompositions: list[DecompositionRecord] = Field(default_factory=list)
    images: list[ImageInfo] = Field(default_factory=list)

    def cell(self, layer: int, position: int) -> CellRecord | None:
        for record in self.cells:
            if record.layer == layer and record.position == position:
                return record
        return None
