# Limitations

OpenJSpace is a research and educational tool. Its readouts are **approximate**
and easy to over-interpret. This page lists the limitations that matter for
drawing (or refusing to draw) conclusions. A short form is surfaced in the UI
and embedded in every exported report.

## Methodological limitations

- **Linearization is approximate.** The Jacobian `J_l` is a first-order
  (linear) approximation of how a residual perturbation propagates to the final
  layer. Real transformer computation is highly nonlinear; the lens captures
  only the linear, on-average part.

- **The Jacobian is corpus-averaged.** `J_l` is an expectation over a fitting
  corpus, source positions, and target positions. It is not exact for any single
  prompt or position, and it inherits whatever distributional biases the fitting
  corpus has. A lens fit on one domain may read out poorly on another.

- **Readouts are tokenizer-bound.** Concepts are expressed only as vocabulary
  tokens. A concept the tokenizer does not represent as a clean token is
  effectively invisible, and the same word can be split across several tokens.

- **Multi-token concepts are poorly represented.** The lens reads out one token
  at a time. Concepts that are inherently multi-token (names, numbers written as
  digits, phrases) are weakly and unreliably surfaced. CLI `--track` and UI
  pinning warn when a tracked concept is multi-token.

- **Top-ranked tokens may be redundant or misleading.** Nearby subword variants,
  whitespace/punctuation tokens, and corpus-frequency artifacts can dominate the
  top of a readout. (The bundled small demo lens shows underscore-filler tokens
  at some layers — see `examples/RESULTS.md`.) The UI defaults to rankings,
  offers whitespace/punctuation filters and variant collapsing, and always
  preserves the raw tokenizer output.

- **J-space decomposition is non-unique.** The J-lens dictionary is correlated
  and overcomplete, so the sparse non-negative decomposition returns *one*
  plausible representative, not a canonical answer. Different runs or budgets
  can yield different, comparably-good decompositions. Reconstruction error and
  explained-norm fraction quantify fit quality but not uniqueness.

- **Most computation may lie outside the readable component.** The lens sees the
  verbalizable projection of an activation onto the vocabulary head. A large
  fraction of the residual stream's role — routing, positional bookkeeping,
  features with no clean vocabulary direction — is not readable this way. A low
  explained-norm fraction is common and expected.

## VLM-specific limitations

- **VLM readouts use the text vocabulary.** For a supported VLM, OpenJSpace
  reads the *language-decoder* residual stream through the *text* unembedding.
  An image-position readout answers "which text concepts is this visual-token
  activation disposed to influence the decoder to verbalize?" — it does **not**
  decode the vision encoder's full representation.

- **VLM patch mappings may be lossy.** Connectors that shuffle or resample
  visual patches (e.g. SmolVLM's pixel-shuffle) break the one-to-one map between
  decoder image tokens and image pixels. OpenJSpace labels patch mapping as
  `exact`, `approximate`, or `unavailable`, surfaces the status in the UI and
  reports, and never fabricates pixel coordinates it does not have. See
  [`VLM_SUPPORT.md`](VLM_SUPPORT.md).

## Engineering limitations (v0.1)

- **Quantized fitting is unsupported.** Gradients through dequantization kernels
  are unreliable, so quantized checkpoints are rejected for fitting with a clear
  error. Documented as experimental future work.

- **Not every architecture is supported.** Only the families in
  `openjspace models list` are handled, and only those marked *tested* have
  passed an integration test against real weights.

- **No distributed fitting, no hosted multi-user service.** A simple in-process
  job queue is sufficient for the MVP; large-scale infrastructure is out of
  scope.

## Interpretation limits — read this before drawing conclusions

- **Concept readouts are not proof of consciousness, intent, belief, or
  deception.** A token appearing high in a readout means an activation has a
  component aligned with that token's unembedding direction, on average, under a
  linear approximation. Nothing more.

- **Causal claims require interventions, not visualization alone.** Observing a
  concept in a readout does not show that the model *uses* it. Establishing that
  requires interventional experiments (ablations, patching, steering), which
  this tool does not perform.

- **Avoid anthropomorphic framing.** OpenJSpace intentionally uses terms like
  *Jacobian-lens readout*, *verbalizable activation directions*, *approximate
  concept readout*, and *J-space proxy*, and avoids "the model is thinking",
  "true thoughts", "lie detector", or "hidden reasoning". Please do the same
  when reporting results built on it.
