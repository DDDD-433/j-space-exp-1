# Contributing to OpenJSpace

Thanks for your interest. OpenJSpace aims to be a scientifically careful,
reproducible interpretability tool, so contributions are held to that standard.

## Development setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,datasets]"
cd web && npm install && npm run build && cd ..   # optional, for the UI
```

Run `openjspace doctor` to confirm your devices are detected.

## Before you open a PR

Run the full local check (also see the `Makefile`):

```bash
make lint        # ruff check src tests examples
make typecheck   # mypy
make test        # unit tests (integration tests are excluded by default)
```

All three must pass. Integration tests (`make test-integration`) download real
weights and are optional locally, but if you touch an adapter, run the ones for
that model.

## Ground rules

- **Correctness and reproducibility first.** Deterministic tests, seeded
  fitting, atomic artifact writes, and validation of all loaded tensors and
  metadata are non-negotiable.
- **Don't replace the upstream methodology with a "simpler" approximation**
  unless it is explicitly named, optional, covered by tests, and documented as
  differing. The production Jacobian is computed by autograd; finite differences
  belong only in tiny unit tests.
- **Precise, non-anthropomorphic language.** Use the terminology in
  `docs/LIMITATIONS.md` ("Jacobian-lens readout", "verbalizable activation
  directions", …). Avoid "thoughts", "lie detector", etc., in code, docs, and
  UI.
- **Typed public interfaces, docstrings on mathematical functions, small
  modules.** No hidden global model state, no hard-coded local paths, no API
  keys, no telemetry. Local-first defaults.
- **No broad `except Exception`** without re-raising with context.

## Adding a model adapter

1. Implement `LensModelAdapter` (see `src/openjspace/models/protocol.py`).
   Reuse `HFDecoderAdapter` when the layout allows.
2. Declare the residual location explicitly and keep it consistent across
   layers; it is stored in lens metadata for compatibility checks.
3. For VLMs: use the official processor/chat template, classify positions by
   modality, and report patch mapping as `exact`/`approximate`/`unavailable` —
   never fabricate geometry.
4. Add unit tests (weightless where possible) and an integration test marked
   `@pytest.mark.integration`.
5. Only change a family's status to **tested** in
   `src/openjspace/models/registry.py` once its integration test passes.

## Reporting qualitative results

Include failures and ambiguous readouts, not only successes (see
`examples/RESULTS.md`). Cherry-picked results are misleading for an approximate
method.

## License

By contributing you agree your contributions are licensed under Apache-2.0.
Preserve attribution headers on files derived from `anthropics/jacobian-lens`.
