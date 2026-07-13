# Methodology

This document gives the precise mathematical definition of the Jacobian lens as
implemented in OpenJSpace, the estimator, the readout, and the sparse J-space
decomposition. It follows the methodology of Anthropic's
[`anthropics/jacobian-lens`](https://github.com/anthropics/jacobian-lens) and
the companion paper *Verbalizable Representations Form a Global Workspace in
Language Models* (transformer-circuits.pub/2026/workspace). Where OpenJSpace
makes an explicit choice or approximation, it is named as such.

## 1. Setup and notation

Consider a decoder-only transformer with a residual stream. Write `h_{l,t}` for
the residual-stream activation at layer `l` and token position `t`. Layers read
from and write to this shared stream; `l = 0` is the earliest residual we fit
at and `l = L` the final residual layer, after which the model applies a final
normalization `Norm` and an unembedding `W_U` to produce logits.

OpenJSpace captures a single, consistent residual location across all layers —
the **block output** (the full residual stream after each decoder block) — and
records that location in the lens artifact so an incompatible lens cannot be
silently applied. See `openjspace.types.ResidualLocation`.

## 2. The Jacobian lens

The Jacobian lens transports an intermediate activation into the final-layer
basis using the **average input–output Jacobian** of the residual stream:

```
J_l = E_{x, t, t' >= t} [ ∂h_{L,t'} / ∂h_{l,t} ]                      (1)
```

where

- `x` is a prompt sampled from a fitting corpus,
- `t` ranges over valid source positions,
- `t'` ranges over the current and future valid target positions (`t' >= t`),
- the expectation averages over prompts, source positions, and valid target
  positions.

`J_l` is a `hidden × hidden` matrix. It is a *first-order, corpus-averaged*
summary of how a perturbation to the residual at `(l, t)` propagates to the
final residual at current-and-future positions. It is **not** exact for any
single input — it is an average linearization.

### Readout

Given `J_l`, the lens reads an activation out through the model's own head:

```
z_{l,t} = W_U · Norm(J_l · h_{l,t})                                   (2)
```

`z_{l,t}` is a vector of vocabulary logits. The UI shows **rankings** or a
display normalization of these logits, never calibrated probabilities: `Norm`
followed by `W_U` reproduces the model's readout machinery, but a transported
intermediate activation is off the manifold the head was trained on, so
absolute magnitudes are not meaningful.

### Logit lens baseline

The **logit lens** is the special case `J_l = I`:

```
z^{logit}_{l,t} = W_U · Norm(h_{l,t}).
```

OpenJSpace computes both readouts plus the model's actual final-layer output for
every inspected cell, and reports top-K overlap and Spearman rank correlation
between them (`openjspace.analysis.comparisons`). Neither lens is treated as
canonically correct.

## 3. The estimator

Directly forming `∂h_{L,t'} / ∂h_{l,t}` for all `t'` is a full Jacobian per
position pair. The estimator OpenJSpace uses (ported from upstream) exploits
decoder causality to compute it efficiently, and is the **only** Jacobian
computation used in production. Finite differences are used *only* in tiny unit
tests as a cross-check.

For one prompt:

1. Run **one forward pass**, with the prompt replicated `dim_batch` times along
   the batch axis and `use_cache=False`. Hooks capture the residual at the
   requested source layers and the target layer. The captured source activation
   at the earliest source layer is marked `requires_grad_(True)` so — with all
   model parameters frozen — the retained autograd graph spans exactly the
   blocks from that layer onward.
2. For each chunk of `dim_batch` output dimensions, build a cotangent that is
   one-hot at output dimension `dim_start + b` (for batch element `b`) **at
   every valid target position simultaneously**, and backpropagate. By
   causality, the gradient arriving at source position `p` equals
   `Σ_{p' >= p} ∂h_L[p'] / ∂h_l[p]` — the sum over current and future target
   positions, which is exactly the inner `t' >= t` sum in (1).
3. Average the resulting rows over the valid source positions. This yields
   `dim_batch` rows of `J_l` per backward pass; `ceil(hidden / dim_batch)`
   passes give the whole matrix.

Across prompts, per-prompt matrices are accumulated as a **float32 running sum**
and divided by the number of contributing prompts at the end. Cost per prompt:
one forward + `ceil(hidden / dim_batch)` backward passes; `dim_batch` trades
memory (the prompt is replicated that many times) for the number of passes, not
total FLOPs.

### Valid positions

`valid_position_mask` (in `openjspace.core.fitting`) excludes:

- the **first `skip_first` positions** (default 16): early positions behave as
  attention sinks and have atypical residual statistics;
- the **final position**: it has no next-token target.

`skip_first` is recorded in metadata and must match when merging shards.

### Sharding and merging

Because the outer expectation in (1) is over prompts, a lens can be fit on
disjoint prompt shards and merged. The merge weights each shard by its
**accumulated prompt count** (`number_of_prompts`), not by a naïve equal average
of shard files, so the result equals a single fit over the union of shards. The
merge first validates that shards agree on model identity, shapes, source
layers, target layer, residual location, and fitting hyperparameters
(`JacobianLens.merge`).

### Numerical safety

Per-prompt gradients are checked for non-finiteness and the fit aborts with a
clear error if NaN/Inf appear (typically a dtype/device problem). Lens tensors
are re-validated for finiteness on save and on load. Between prompts, references
are released, `gc.collect()` runs, and the device cache is cleared.

## 4. Sparse J-space decomposition

Beyond top-K readout, OpenJSpace can approximate an activation as a **sparse,
non-negative** combination of J-lens vocabulary directions. Define the linear
J-lens vectors at a layer as the rows of `W_U J_l`:

```
v_t = (W_U J_l)[t]   ∈ R^{hidden}.
```

Given an activation `h`, approximately solve

```
min_{a >= 0, |S| <= K}  || h - Σ_{i in S} a_i v_i ||_2 .              (3)
```

**Why omit `Norm` here.** Pre-softmax logits are determined, approximately and
up to a data-dependent scale, by the inner products `<v_t, h>` with the *linear*
J-lens vectors. Using the linear vectors keeps the dictionary a fixed linear map
(so (3) is a genuine sparse-coding problem) and matches the paper's
probe/decomposition usage. This is a deliberate, documented approximation
(`openjspace.core.normalization`): the full nonlinear readout (2) still uses
`Norm`.

**Solver.** OpenJSpace uses non-negative orthogonal matching pursuit (NN-OMP), a
greedy pursuit in the same family as the paper's gradient pursuit
(`openjspace.core.decomposition`):

1. Select the atom with the largest positive correlation to the current
   residual (correlations use unit-normalized atoms so scale differences don't
   bias selection).
2. Refit **all** active coefficients by non-negative least squares (projected
   gradient with a Lipschitz step) on the active set.
3. Repeat until `K` atoms are selected, the residual falls below tolerance, or
   no atom has positive correlation. Atoms whose refit coefficient collapses to
   zero are dropped.

Reported metrics: selected token concepts, non-negative coefficients,
reconstruction error `||h - ĥ||`, residual norm, and explained-norm fraction
`1 - (residual / ||h||)^2`.

**Non-uniqueness.** The dictionary (`W_U J_l` rows, `vocab_size` of them into a
`hidden`-dimensional space) is **correlated and overcomplete**, so (3) has no
unique solution: the result is *one* sparse representative, not a canonical
decomposition. Every decomposition carries this warning. Top-K by inner product
is **not** a sparse decomposition and is never labeled as one.

## 5. What this does and does not measure

The lens is a linear, corpus-averaged, tokenizer-bound projection onto the
model's vocabulary head. It measures the **verbalizable component** of a
residual activation — the part that lines up, on average, with directions the
model would use to produce tokens. It does not measure the full state, does not
establish causation, and does not decode non-verbalizable computation. See
[`LIMITATIONS.md`](LIMITATIONS.md).
