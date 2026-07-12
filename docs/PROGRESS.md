# Progress

Status of the v0.1 milestones. Updated as work lands.

## Milestones

| # | Milestone | Status |
| --- | --- | --- |
| 0 | Research & design (upstream inspection, `DESIGN.md`, licensing, artifact schema, model protocol) | done |
| 1 | Text core (scaffold, tiny + causal-LM adapters, fitting, applying, serialization, CLI, math tests) | done |
| 2 | Visualizer (FastAPI, in-process jobs, React UI, layer × position grid, concept details, rank tracking, JSON + HTML export) | done |
| 3 | First real model (Qwen adapter, integration test, committed demo lens, example reports) | done |
| 4 | VLM (SmolVLM adapter, modality mapping, image-token visualization, approximate patch mapping, VLM tests + example) | done |
| 5 | J-space decomposition (NN-OMP, coefficients + reconstruction metrics, CLI + UI integration, tests) | done |
| 6 | Polish (README, METHODOLOGY/LIMITATIONS/VLM_SUPPORT docs, CI, lint, type check, reproducible demo, this page) | done |

## What works

- `openjspace doctor`, `models list`, `model inspect`, `fit`, `merge`,
  `inspect`, `decompose`, `export`, `serve` (all with `--help`).
- Tiny CPU test model fits a lens in unit tests (no downloads).
- Real Qwen2.5-0.5B: integration-tested fit → save → load → inspect, with the
  model-output row matching the model's own logits; a demo lens is committed
  under `artifacts/qwen2.5-0.5b-instruct-demo`.
- Text prompts produce a layer × position J-lens grid; any cell is inspectable
  for top-K concepts; J-lens vs. logit-lens vs. model-output comparison with
  overlap and rank-correlation metrics.
- Runs export to valid JSON and self-contained HTML (round-trip tested).
- SmolVLM-256M accepts an image + prompt; decoder positions are classified by
  modality; image-token readouts are inspectable across layers; patch mapping is
  reported as `approximate` and never fabricated.
- Sparse non-negative J-space decomposition (CLI + UI) with reconstruction
  error, residual norm, explained-norm fraction, and a non-uniqueness warning.
- Scientific limitations surfaced in the README, the UI, and every report.

## Testing

- Unit tests (`pytest -m "not integration"`) are deterministic and need no
  downloads: math (autograd-vs-finite-difference on a tiny net, masking, skipped
  prefix, count-weighted merge, NaN detection, determinism), adapters,
  serialization, decomposition, VLM mapping, and API.
- Integration tests (`pytest -m integration`) download real weights:
  `tests/test_integration_qwen.py` (tiny-Qwen2 fast path + Qwen2.5-0.5B slow
  path) and `tests/test_integration_smolvlm.py` (SmolVLM-256M). Documented as
  optional; run in CI is the unit suite only.
- CI (`.github/workflows/tests.yml`): ruff lint + format check, mypy, unit
  tests on Python 3.11/3.12 (CPU torch), and a web build.

## Non-goals for v0.1 (intentionally not implemented)

Support for every HF architecture; distributed fitting; hosted multi-user
service; arbitrary remote-code models without review; training an oracle lens;
full multi-token template lenses; vision-encoder J-lenses without a justified
output basis; large-scale causal interventions; automated safety conclusions;
quantized Jacobian fitting; mobile inference.

## Known rough edges

- The committed demo lens is deliberately under-fit (24 prompts) and shows
  filler-token artifacts at some layers — see `examples/RESULTS.md`. Fit more
  prompts for cleaner readouts.
- SmolVLM full image tiling is slow on CPU; the examples/tests disable image
  splitting (single 8×8 tile) for tractability.
- No recorded UI GIF is committed yet (needs a display); self-contained HTML
  reports under `examples/reports/` stand in for offline viewing.
