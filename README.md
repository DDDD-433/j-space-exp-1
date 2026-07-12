# OpenJSpace

**An interactive Jacobian-lens and J-space visualizer for open-weight language
models and vision-language models — local-first, reproducible, scientifically
careful.**

OpenJSpace lets you fit or load a *Jacobian lens* for an open-weight model, run
a prompt (optionally with an image for supported VLMs), and inspect the
approximate, verbalizable concepts associated with residual-stream activations
across layers and token positions — side by side with the standard logit lens
and the model's own output.

> ### Scientific disclaimer
>
> A Jacobian-lens readout is an **approximate, corpus-averaged, linearized**
> projection of a residual-stream activation, decoded through the model's own
> unembedding. It surfaces **verbalizable activation directions** — *not* a
> model's literal "thoughts", beliefs, intentions, or plans. Scores are
> **rankings**, not calibrated probabilities. Most of a model's computation
> lies **outside** this readable component. Causal claims require interventions,
> not visualization. See [`docs/LIMITATIONS.md`](docs/LIMITATIONS.md), which is
> also surfaced inside the UI and every exported report.

OpenJSpace builds on the methodology of Anthropic's
[`anthropics/jacobian-lens`](https://github.com/anthropics/jacobian-lens)
(Apache-2.0) — see [Attribution](#citation-and-attribution).

---

## Screenshot

The web UI (`openjspace serve`) shows a layer × position grid, per-cell concept
details, rank tracking, a J-lens vs. logit-lens comparison, VLM image tokens,
and run metadata:

<!-- A recorded GIF/screenshot is added under docs/img/ when a display is
available; committed self-contained HTML reports in examples/reports/ show the
exported view offline in the meantime. -->

Committed example reports (open directly in a browser):
`examples/reports/text_qwen2.5_0.5b/*.html`,
`examples/reports/vlm_smolvlm_synthetic/report.html`.

## Installation

Python 3.11+ and PyTorch. From a clone:

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"          # core + dev tools (pytest, ruff, mypy)
pip install -e ".[datasets]"     # optional: stream WikiText for fitting
pip install torchvision          # optional: required only for SmolVLM images
```

The web UI is prebuilt into `web/dist`. To rebuild it (Node 18+):

```bash
cd web && npm install && npm run build
```

## Quick start (one command)

The repository ships a small demo lens for `Qwen/Qwen2.5-0.5B-Instruct`
(`artifacts/qwen2.5-0.5b-instruct-demo`), so you can launch the UI and inspect a
prompt immediately:

```bash
openjspace serve            # then open http://127.0.0.1:8000
```

Enter model `Qwen/Qwen2.5-0.5B-Instruct`, select the bundled lens, type a
prompt, and hit **Run**. (First run downloads the 0.5B weights.)

## Text-model example (CLI)

```bash
openjspace inspect \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --lens artifacts/qwen2.5-0.5b-instruct-demo \
  --prompt "The capital of France is" \
  --positions last:1 --top-k 10 \
  --output runs/paris
# -> runs/paris/run.json  and  runs/paris/report.html
```

With this demo lens, `' Paris'` is already among the top J-lens concepts by
layer 8 — several layers before the output. See
[`examples/RESULTS.md`](examples/RESULTS.md) for a fuller, **non-cherry-picked**
set of successes, ambiguous cases, and failures, or run
`python examples/text_demo.py`.

## VLM example

SmolVLM (Idefics3) reads out the **language-decoder** residual stream after
visual embeddings enter the sequence. Requires `torchvision`.

```bash
python examples/vlm_demo.py --image photo.jpg --prompt "What is in this image?"
```

or in the UI: load `HuggingFaceTB/SmolVLM-256M-Instruct`, upload an image, and
open the **Image Tokens** tab. Image-token readouts answer *"which text
concepts is this visual-token activation disposed to influence the decoder to
verbalize?"* — they do **not** decode the vision encoder directly, and patch
mapping is **approximate** (see [`docs/VLM_SUPPORT.md`](docs/VLM_SUPPORT.md)).

## Fitting your own lens

```bash
openjspace fit \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --dataset wikitext \
  --output artifacts/my-lens \
  --layers 2,4,8,12,16,20 \
  --max-seq-len 128 --num-prompts 100 --dim-batch 8 \
  --device auto --checkpoint-every 10
```

Fits are **resumable** (atomic checkpoints), **deterministic** (seeded prompt
order), and **shardable** — fit disjoint shards with `--shard-index/--num-shards`
and combine them with `openjspace merge` (count-weighted, not a naïve average).
Run `openjspace doctor` first to check devices and likely fitting cost.

## Hardware expectations

| Setup | Fitting | Inspection |
| --- | --- | --- |
| Tiny test model (CI) | milliseconds | milliseconds |
| Qwen2.5-0.5B on CPU | ~1 min per prompt per 6 layers (dim_batch 32) | seconds |
| Qwen2.5-0.5B on CUDA/MPS | seconds per prompt | sub-second |
| ≥7B models | GPU strongly recommended | GPU recommended |

Lens accumulation is always float32; model weights may be lower precision as
long as activation gradients stay finite. Use `--dim-batch` to trade memory for
the same total work. Quantized checkpoints are rejected for fitting (see
Limitations).

## Supported models

Status is honest: a family is **tested** only once an integration test against
real weights has passed.

| Family | Kind | Status | Example |
| --- | --- | --- | --- |
| Qwen2 / Qwen2.5 | text | **tested** | `Qwen/Qwen2.5-0.5B-Instruct` |
| SmolVLM / SmolVLM2 (Idefics3) | vlm | **tested** | `HuggingFaceTB/SmolVLM-256M-Instruct` |
| Qwen3 | text | experimental | `Qwen/Qwen3-0.6B` |
| Llama 3.x | text | experimental | `meta-llama/Llama-3.2-1B` |
| Mistral | text | experimental | `mistralai/Mistral-7B-v0.3` |
| Gemma 2/3 | text | experimental | `google/gemma-2-2b` |
| GPT-2 | text | experimental | `openai-community/gpt2` |
| Pythia / GPT-NeoX | text | experimental | `EleutherAI/pythia-160m` |
| Qwen2.5-VL / Qwen3-VL | vlm | planned | `Qwen/Qwen2.5-VL-3B-Instruct` |

`openjspace models list` prints this table live. *experimental* means the
layout is supported but untested against real weights; *planned* is not yet
implemented.

## Method summary

For a residual activation `h_{l,t}` at source layer `l`, position `t`, estimate
the average input–output Jacobian to the final layer `L`:

```
J_l = E_{x, t, t' >= t} [ ∂h_{L,t'} / ∂h_{l,t} ]
```

(expectation over prompts `x`, source positions `t`, and current-and-future
target positions `t' >= t`). The readout at a layer is:

```
z_{l,t} = W_U · Norm(J_l · h_{l,t})
```

where `Norm` is the model's final normalization and `W_U` its unembedding. The
estimator uses one forward pass per prompt (replicated `dim_batch`× along the
batch axis) and `ceil(hidden / dim_batch)` backward passes, injecting one-hot
cotangents at every valid target position at once. Full derivation and the
exact estimator are in [`docs/METHODOLOGY.md`](docs/METHODOLOGY.md).

## Comparison with the logit lens

The **logit lens** is the special case `J_l = I`: it applies the unembedding
directly to an intermediate activation. The Jacobian lens instead *transports*
the activation into the final-layer basis first, which accounts (to first
order, on average) for the downstream computation the residual still drives.
OpenJSpace always shows both, plus the model's actual output distribution, with
top-K overlap and rank-correlation metrics. **Neither is universally "correct"**
— they are different projections, and the comparison itself is informative.

## Limitations (short form)

Linearization is approximate; the Jacobian is corpus-averaged; readouts are
tokenizer-bound and weak on multi-token concepts; top-ranked tokens can be
redundant or misleading; J-space decomposition is non-unique; most computation
lies outside the readable component; VLM patch mappings can be lossy; and none
of this is evidence of consciousness, intent, belief, or deception. Full detail
in [`docs/LIMITATIONS.md`](docs/LIMITATIONS.md).

## Artifact format

A lens is a directory with `lens.safetensors` (the `J_l` matrices, fp16 storage)
and `metadata.json` (a versioned schema recording model id/revision,
architecture, tokenizer, hidden/vocab/layer sizes, source layers, target layer,
**residual location**, sequence length, skipped-prefix count, prompt/position
counts, dtypes, dataset, and library versions). Before applying a lens,
OpenJSpace validates architecture, sizes, tokenizer, and residual location so an
incompatible lens cannot be silently applied (override with `--force`, warned).

## CLI

```
openjspace doctor                      # environment, devices, fitting compatibility
openjspace models list                 # supported-model table
openjspace model inspect --model ID    # a model's lens-relevant structure
openjspace fit ...                     # fit a lens (resumable, shardable)
openjspace merge SHARDS... --output …  # count-weighted shard merge
openjspace inspect ...                 # apply a lens, save run JSON + HTML
openjspace decompose ...               # sparse non-negative J-space decomposition
openjspace export RUN --format html    # re-export a saved run
openjspace serve                       # local web UI
```

Every command supports `--help`.

## Citation and attribution

OpenJSpace derives its fitting estimator, hooks, tiny-model test strategy, and
lens/merge semantics from Anthropic's `anthropics/jacobian-lens` (Apache-2.0),
the companion code to *Verbalizable Representations Form a Global Workspace in
Language Models* (transformer-circuits.pub/2026/workspace). Derived files carry
attribution headers; see [`NOTICE`](NOTICE) and [`LICENSE`](LICENSE) (Apache-2.0).

If you use OpenJSpace, please also cite the original Jacobian-lens work.

## Roadmap

- Richer VLM adapter (Qwen2.5-VL / Qwen3-VL) with exact patch geometry where available.
- Downloadable pre-fit lenses on the Hugging Face Hub.
- Tuned-lens comparison.
- Optional gradient-pursuit decomposition backend.

Non-goals for 0.1 (deliberately out of scope): every HF architecture,
distributed fitting, hosted multi-user service, quantized fitting, and any
automated safety conclusions. See [`docs/PROGRESS.md`](docs/PROGRESS.md).

## License

Apache-2.0. See [`LICENSE`](LICENSE) and [`NOTICE`](NOTICE).
