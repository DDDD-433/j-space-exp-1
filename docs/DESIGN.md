# OpenJSpace — Design Document

**Status:** living document; written at Milestone 0 after inspecting the upstream
reference implementation, updated as milestones land.

OpenJSpace is a local-first, open-source interactive Jacobian-lens and J-space
visualizer for open-weight language models and vision-language models (VLMs).
It builds on the methodology of Anthropic's
[`anthropics/jacobian-lens`](https://github.com/anthropics/jacobian-lens)
reference implementation (Apache-2.0), the companion code for
*Verbalizable Representations Form a Global Workspace in Language Models*
(transformer-circuits.pub/2026/workspace).

---

## 1. Upstream inspection summary

We inspected the full upstream repository (`jlens` package, tests, walkthrough
notebook) and the relevant paper sections (Jacobian-lens construction, J-lens
readout, sparse J-space decomposition via gradient pursuit, comparison with the
logit lens and tuned lens, and limitations). Findings:

### 1.1 What upstream provides

| Component | File | Notes |
|---|---|---|
| Fitting estimator | `jlens/fitting.py` | One forward per prompt (replicated `dim_batch`× along the batch axis), `ceil(d_model / dim_batch)` backward passes; one-hot cotangents injected at every valid target position at once; gradient at source position `p` is `Σ_{p' ≥ p} ∂h_final[p'] / ∂h_l[p]`, then mean over valid source positions; running mean over prompts; fp32 accumulation; atomic resumable checkpoints; skips prompts too short to leave valid positions. |
| Valid-position mask | `jlens/fitting.py` | Skips the first 16 positions (attention sinks, atypical residual statistics) and the final position (no next-token target). |
| Hooks | `jlens/hooks.py` | `ActivationRecorder`: forward hooks on residual blocks; records **block output** (tuple-unwrapping HF block outputs); `start_graph_at` marks the earliest source activation `requires_grad_(True)` so the retained graph spans only the blocks from that layer onward (all parameters are frozen). |
| Lens object | `jlens/lens.py` | `JacobianLens`: `{layer: J_l}` dict, `transport` (`h @ J_l.T`), `apply` (forward + readout, with `use_jacobian=False` giving the logit-lens baseline), `merge` (n_prompts-weighted mean), save/load via `torch.save` (fp16 storage), `from_pretrained` for HF Hub. |
| Model protocol | `jlens/protocol.py` | Minimal: `n_layers`, `d_model`, `layers`, `tokenizer`, `encode`, `forward`, `unembed`. |
| HF adapter | `jlens/hf.py` | `Layout` dataclass (dotted attr paths) + auto-detection across Llama/Qwen/Mistral/Gemma/Phi/GPT-2/GPT-NeoX layouts; freezes params; `unembed` = final norm + lm_head (+ optional logit softcapping); `use_cache=False` forward. |
| Visualization | `jlens/vis.py` | `compute_slice` (position × layer top-K grid, tracked-token full-vocab ranks computed chunked), self-contained HTML page (d3 inlined, gzip+base64 typed-array payload) or fetch-mode sidecar files. |
| Tests | `tests/` | Tiny 4-layer CPU decoder (`h + 0.1·W·h` blocks) with analytically checkable Jacobians (`J_{L-1} = I + W_L` exactly); mask/merge/resume/negative-index/error-path tests; mock-HF layout tests; rank-computation tests. |
| Examples | `jlens/examples.py` | Qualitative prompt suite (multi-hop, modulation, ASCII face, code bug, etc.), WikiText-103 fitting-corpus loader. |

### 1.2 Key mathematical facts (from paper + code)

- **Estimator.** `J_l = E[∂h_{L,t'} / ∂h_{l,t}]`, expectation over prompts,
  source positions `t`, and current-and-future target positions `t' ≥ t`.
  Implementation injects a one-hot cotangent (in one output dimension) at *all*
  valid target positions simultaneously; by causality of the decoder,
  `∂h_L[t'] / ∂h_l[t] = 0` for `t' < t`, so a single backward yields
  `Σ_{t' ≥ t}` at each source `t` for free. Mean over valid source positions,
  then mean over prompts. This is the paper's reduction; a strict per-position
  estimator (`t' = t` only) is a documented variant.
- **Readout.** `lens_l(h) = W_U · Norm(J_l · h)` where `Norm` is the model's
  own final normalization and `W_U` its unembedding (upstream folds
  `Norm`+`W_U` into `unembed`). Scores are *rankings*, not calibrated
  probabilities. Logit lens is the `J_l = I` special case.
- **Corpus.** Paper default: 1000 sequences of 128 tokens from a
  pretraining-like distribution; quality beats logit/tuned lens with as few as
  ~10 prompts and saturates quickly (~100 prompts usable). First 16 positions
  skipped; final position skipped.
- **J-space / sparse decomposition.** The J-space at sparsity `k` is the set of
  non-negative combinations of ≤ `k` rows of `W_U J_l` (J-lens vectors,
  pulled back to layer-`l` residual space). Paper operationalizes membership
  via **gradient pursuit** (Blumensath & Davies 2008 family), `k` ≤ ~25,
  typically 16. The dictionary is overcomplete and correlated, so the
  decomposition is non-unique; it is a different (less redundant) inventory
  than top-k by inner product. J-space components carry only ~6–15% of
  activation variance in the paper's measurements.
- **Target layer.** Defaults to the final layer; the penultimate layer can be
  better-conditioned (upstream exposes `target_layer`).
- **Diagnostics.** Per-prompt Jacobian norm (`max_l ‖J‖_F / √d`) flags
  heavy-tailed outliers; the relative shift of the running mean (~1/n once
  settled) tracks convergence.

### 1.3 Licensing

Upstream code and data are Apache-2.0 (`Copyright 2026 Anthropic PBC`,
SPDX headers on every file). We may reuse and adapt with attribution:

- OpenJSpace is itself Apache-2.0.
- A `NOTICE` file records that portions of `openjspace.core` (fitting
  estimator, hooks, merge semantics) and the tiny test model are derived from
  `anthropics/jacobian-lens`.
- Files containing derived code carry a header noting the derivation and
  preserving the upstream copyright line alongside ours.

## 2. What we reuse vs. extend vs. build new

### Reused (adapted with attribution)

- The **fitting estimator** (cotangent batching, valid-position mask,
  running-mean accumulation, atomic checkpoint/resume, skip-and-don't-recount
  semantics) — ported nearly verbatim into `openjspace/core/fitting.py`; this
  is the scientifically load-bearing part and we deliberately do not
  "simplify" it.
- The **hook strategy** (`ActivationRecorder` with `start_graph_at` graph
  rooting) — `openjspace/core/hooks.py`.
- The **count-weighted merge** semantics.
- The **tiny analytic test model** idea (residual blocks `h + 0.1·W·h` with an
  exactly checkable `J_{L-1} = I + W_L`) — `openjspace/models/tiny.py`.
- The HF **layout table** approach for locating decoder internals.

### Extended

- **Serialization**: upstream uses `torch.save` with minimal metadata. We
  define a versioned artifact format — `lens.safetensors` + `metadata.json` —
  with model/tokenizer identity, revision, residual location, fitting
  hyperparameters, and library versions, plus validation-before-apply and an
  explicit `--force` override (§5).
- **Model protocol**: upstream's protocol is minimal (`encode`/`forward`/
  `unembed`). We extend it with position metadata (`classify_positions`),
  explicit residual-location declaration, multimodal `prepare_inputs`, and
  introspection accessors (final norm, unembedding weight) needed by the
  decomposition module and validation (§4).
- **Application**: upstream's `apply` returns raw logit tensors. We add a
  structured inspection pipeline (top-K per layer × position, activation
  norms, rank comparisons vs. logit lens and final logits, pinned-concept
  tracking) with a JSON-serializable schema for UI/export.
- **Visualization**: upstream ships one d3 slice page. We build a React +
  TypeScript + Vite web app served by FastAPI, plus a self-contained HTML
  export (no CDN dependency).

### New

- **VLM support** (`openjspace/models/smolvlm.py`, `qwen_vl.py`): residual
  recording in the language decoder of a VLM after visual embeddings are
  merged; position modality classification; patch-geometry mapping with
  explicit `exact / approximate / unavailable` status.
- **Sparse non-negative decomposition** (`openjspace/core/decomposition.py`):
  non-negative orthogonal matching pursuit with least-squares refit
  (equivalently, a greedy non-negative pursuit in the gradient-pursuit
  family), reporting coefficients, reconstruction error, explained norm
  fraction, and a non-uniqueness warning. Top-k-by-inner-product is *never*
  labeled a decomposition.
- **CLI** (Typer): `doctor`, `models list`, `model inspect`, `fit`, `merge`,
  `inspect`, `decompose`, `serve`, `export`.
- **Server** (FastAPI): in-process job queue (fit jobs are long-running),
  run artifact storage on the local filesystem, path validation.

## 3. Architectural decisions

1. **Hooks, not `output_hidden_states`.** Fitting needs *differentiable*
   tensors at exact residual locations; `output_hidden_states=True` returns
   detached copies in some code paths and does not let us root the autograd
   graph at the earliest source layer. We hook block outputs.
2. **Residual location = `block_output`** for every adapter in v0.1, matching
   upstream. The location string is stored in lens metadata; applying a lens
   whose location differs from the adapter's is an error (no silent
   mismatches).
3. **Adapters wrap already-loaded HF models.** Loading (device, dtype,
   revision) is a separate concern (`registry.load_model`) so tests can inject
   mocks and the server can manage model lifetime.
4. **fp32 accumulation, fp16 storage.** Jacobians accumulate in float32 on
   CPU; artifacts store float16 (entries are O(1); fp16 mantissa beats bf16
   here — upstream's reasoning, kept). Application casts back to fp32.
5. **Safetensors for artifacts.** One `lens.safetensors` (keys `J_{layer}`)
   plus `metadata.json`. Checkpoints (running sums) also use safetensors +
   JSON sidecar; `torch.save` is avoided everywhere but never loaded without
   `weights_only=True`.
6. **Vocabulary-sized outputs are never stored by default.** Runs store top-K
   ids/strings/scores and ranks of pinned/tracked tokens; `--store-full-logits`
   opts into full tensors.
7. **In-process job queue.** Fitting jobs run on a single background worker
   thread with cooperative cancellation and progress counters; no external
   queue/database. Artifacts and runs live under a configurable
   `OPENJSPACE_HOME` (default `~/.openjspace`, overridable per-invocation).
8. **Self-contained HTML export** renders the same JSON run schema the React
   app consumes, using an inlined template (no CDN fetch at view time).
9. **Monorepo layout** as specified in the task, with `web/` (Vite React TS)
   compiled to static assets served by FastAPI in `serve` mode.

## 4. Model protocol (summary)

```python
class LensModelAdapter(Protocol):
    model_id: str
    n_layers: int            # residual blocks in the *language* decoder
    hidden_size: int
    vocab_size: int
    residual_location: ResidualLocation   # "block_output" in v0.1
    def tokenize_text(text, max_length) -> TokenizedInput: ...
    def prepare_inputs(prompt, images=None, max_length=...) -> ModelInputs: ...
    def get_residual_modules() -> Sequence[nn.Module]: ...
    def get_final_norm() -> nn.Module: ...
    def get_unembedding_weight() -> torch.Tensor: ...
    def decode_token_ids(ids) -> list[str]: ...
    def classify_positions(inputs) -> list[PositionMetadata]: ...
    def forward_with_activations(inputs, layers, grad_from=None) -> ForwardResult: ...
    def unembed(residual) -> torch.Tensor: ...   # final norm + W_U (+softcap)
```

`forward_with_activations` is the single forward-pass entry point used by both
fitting (grad enabled, graph rooted at `grad_from`) and inspection (inference
mode). `classify_positions` returns per-position modality metadata; for text
models everything is `text`/`special`.

## 5. Lens artifact format (v1)

```
artifact_dir/
├── lens.safetensors      # J_{l}: [hidden, hidden] fp16, one key per source layer
└── metadata.json
```

`metadata.json` fields: `format_version=1`, `method="jacobian_lens"`,
`model_id`, `model_revision`, `model_architecture`, `tokenizer_id`,
`tokenizer_revision`, `hidden_size`, `vocab_size`, `n_layers`,
`source_layers`, `target_layer`, `residual_location`, `sequence_length`,
`skip_first_positions`, `number_of_prompts`, `number_of_valid_positions`,
`dtype_used_for_model`, `dtype_used_for_accumulation="float32"`,
`fitting_dataset`, `created_at`, `library_versions`, `notes`.

Validation before apply: architecture, hidden size, vocab size, tokenizer id,
layer count, residual location, and (when recorded on both sides) model
revision. Mismatch → error listing every failed check; `--force`/`force=True`
downgrades to a prominent warning.

Merging weights each shard by `number_of_valid_positions × number_of_prompts`
proxy — concretely by the shard's accumulated **prompt count** (matching
upstream `merge`) *and* validates metadata compatibility first. (Upstream
weights by `n_prompts`; positions-per-prompt vary, so we additionally record
total valid positions and use prompt-count weighting for the mean over
prompts, which is the estimator's outer expectation. This matches the
upstream estimator exactly.)

## 6. Mathematical assumptions (explicit)

1. **Linearization is approximate.** `J_l` is a corpus-averaged first-order
   approximation of layers `l→L`; per-prompt deviations are unbounded.
2. **The average is corpus-dependent.** A lens fitted on web text encodes
   web-text statistics; readouts on far-out-of-distribution prompts (or chat
   templates unseen in fitting) may degrade.
3. **Readouts are tokenizer-bound.** Concepts without single-token names
   surface diffusely or not at all.
4. **Scores are not probabilities.** We display ranks and normalized scores;
   softmax over lens logits is *not* a calibrated distribution.
5. **Decomposition is non-unique.** The J-lens dictionary is overcomplete and
   correlated; greedy pursuit returns *a* sparse representative, not *the*
   decomposition.
6. **Causality requires interventions.** Visualization alone shows
   dispositions, not mechanisms; we make no causal or safety claims from
   readouts.

## 7. VLM-specific limitations

- We record residuals **only in the language decoder**, after projected visual
  embeddings enter the sequence. The vision encoder itself is out of scope for
  v0.1 (a vision-encoder J-lens has no justified output basis).
- Image-position readouts answer: *which text concepts is this visual-token
  activation disposed to make the decoder verbalize?* They do not decode the
  vision encoder's full representation.
- Patch geometry: SmolVLM-style pixel-shuffle connectors merge patches
  (mapping `approximate`); resampler architectures (e.g. Idefics2-style
  perceiver) destroy locality (mapping `unavailable`). The status is recorded
  per run and shown in the UI; we never fabricate pixel coordinates.
- Chat templates and image splitting must use the official processor; we
  preserve the exact multimodal sequence and classify positions from the
  processor output (image token ids, boundary tokens, specials).
- The fitting corpus for v0.1 VLM lenses is **text-only** prompts through the
  language decoder. A text-fitted lens applied at image positions is an
  additional distribution shift; documented in LIMITATIONS.md and surfaced in
  the UI as a caveat.

## 8. Implementation phases

- **M0 (this doc):** upstream inspection, licensing, schema + protocol design.
- **M1 — text core:** package scaffold; `core/` (hooks, fitting, applying,
  normalization, serialization); tiny test adapter; generic HF decoder
  adapter; CLI (`doctor`, `fit`, `merge`, `inspect`, `models`, `export`,
  `decompose` stub); mathematical tests (finite-difference check on a tiny
  linear net, masking, skip-prefix, merge weighting, determinism, NaN
  detection).
- **M2 — visualizer:** FastAPI app + job queue; run schema; React UI (Lens
  Grid, Concept Explorer, Rank Tracking, Lens Comparison, Image Tokens,
  Metadata tabs); JSON + self-contained HTML export.
- **M3 — first real model:** Qwen2.5 adapter entry (thin over HF decoder
  adapter), optional integration test (`-m integration`), example fitted-lens
  workflow + example reports.
- **M4 — VLM:** SmolVLM adapter (lightweight; Idefics3 architecture),
  modality mapping + patch metadata, image upload in server/UI, VLM tests with
  synthetic images (mock-based in CI, real-model integration optional).
- **M5 — J-space decomposition:** non-negative OMP/gradient pursuit,
  reconstruction metrics, CLI + UI integration, recovery tests.
- **M6 — polish:** README, METHODOLOGY, VLM_SUPPORT, LIMITATIONS, PROGRESS;
  CI workflow (ruff + mypy + pytest); screenshots; release checklist.

## 9. Dependency plan

Python ≥ 3.11 (developed on 3.12): `torch`, `transformers`, `safetensors`,
`huggingface_hub`, `numpy`, `pydantic` (v2), `typer`, `fastapi`, `uvicorn`,
`pillow` (VLM), `python-multipart` (uploads). Dev: `pytest`, `ruff`, `mypy`,
`httpx` (API tests). Web: `react`, `react-dom`, `typescript`, `vite` — charts
rendered with hand-rolled SVG (no heavy chart dependency).

## 10. Non-goals for v0.1

Every-architecture support, distributed fitting, hosted multi-user service,
remote-code models without review, oracle/template lenses, vision-encoder
lenses, large-scale interventions, automated safety conclusions, quantized
fitting (documented as experimental future work), mobile.
