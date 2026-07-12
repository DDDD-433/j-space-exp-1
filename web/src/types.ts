// Mirrors src/openjspace/report/schema.py (RunResult) and server schemas.

export interface ConceptEntry {
  token_id: number;
  token_text: string;
  token_display: string;
  score: number;
  normalized_score: number;
}

export type Modality = "text" | "image_token" | "image_boundary" | "special" | "unknown";

export interface PositionInfo {
  index: number;
  modality: Modality;
  token_id: number | null;
  token_text: string | null;
  image_index: number | null;
  patch_index: number | null;
  patch_row: number | null;
  patch_col: number | null;
}

export interface CellRecord {
  layer: number;
  position: number;
  jlens_top: ConceptEntry[];
  logit_lens_top: ConceptEntry[];
  comparison: Record<string, number>;
  activation_norm: number;
  transported_norm: number;
  is_model_output: boolean;
}

export interface TrackedConcept {
  token_id: number;
  token_text: string;
  token_display: string;
  ranks: number[][];
  scores: number[][];
}

export interface DecompositionEntry {
  token_id: number;
  token_text: string;
  token_display: string;
  coefficient: number;
}

export interface DecompositionRecord {
  layer: number;
  position: number;
  entries: DecompositionEntry[];
  k_requested: number;
  reconstruction_error: number;
  residual_norm: number;
  explained_norm_fraction: number;
  n_iterations: number;
  warning: string;
}

export interface ImageInfo {
  image_index: number;
  filename: string;
  width: number;
  height: number;
  grid_rows: number | null;
  grid_cols: number | null;
  data_uri: string;
}

export interface RunMetadata {
  run_id: string;
  created_at: string;
  model_id: string;
  model_architecture: string;
  model_kind: "text" | "vlm";
  device: string;
  dtype: string;
  lens_path: string;
  lens_metadata: Record<string, unknown>;
  prompt: string;
  used_chat_template: boolean;
  layers: number[];
  top_k: number;
  n_positions: number;
  patch_mapping: "exact" | "approximate" | "unavailable";
  disclaimer: string;
  warnings: string[];
}

export interface RunResult {
  schema_version: number;
  metadata: RunMetadata;
  positions: PositionInfo[];
  cells: CellRecord[];
  tracked: TrackedConcept[];
  decompositions: DecompositionRecord[];
  images: ImageInfo[];
}

export interface ModelInfo {
  model_id: string;
  architecture: string;
  kind: "text" | "vlm";
  status: string;
  n_layers: number;
  hidden_size: number;
  vocab_size: number;
  residual_location: string;
  tokenizer_id: string;
  device: string;
  dtype: string;
}

export interface ArtifactInfo {
  name: string;
  path: string;
  metadata: Record<string, unknown>;
}

export interface UploadInfo {
  name: string;
  width: number;
  height: number;
}

export interface SelectedCell {
  layer: number;
  position: number;
}
