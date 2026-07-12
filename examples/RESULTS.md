# Example results (qualitative, not cherry-picked)

These are readouts from the bundled demo lens on **Qwen/Qwen2.5-0.5B-Instruct**.
They are included to show what the tool produces on a *deliberately small,
cheap* lens — **not** to claim the method works well here. Failures and
ambiguous readouts are reported alongside the successes, as required by the
project's evaluation policy.

## Demo lens

| Property | Value |
| --- | --- |
| Model | `Qwen/Qwen2.5-0.5B-Instruct` (24 layers, hidden 896, vocab 151936) |
| Source layers | 2, 4, 8, 12, 16, 20 (target = final layer 23) |
| Residual location | `block_output` |
| Fitting corpus | WikiText-103, 24 prompts, seq len 64, skip first 16 |
| Accumulation | float32 |

**This lens is under-fit on purpose** (24 prompts vs. the paper's ~1000) so it
runs on a CPU in ~30 minutes. Treat every readout below as a hypothesis to
inspect in the UI, never as ground truth. Full self-contained reports are in
`examples/reports/text_qwen2.5_0.5b/*.html`.

Prompts are read out at the **last token**; the row marked *output* is the
model's own final-layer distribution (`J = I`), not a lens estimate.

## Successes (concept surfaces before the output)

- **`capital-france`** — "The capital of France is". By layer 8 the top J-lens
  concepts already include `' Paris'`, `' French'`, `' city'`; layer 12 puts
  `' Paris'` first (ahead of `' Marseille'`). The model output is `' Paris'`.
  A clean example of the target concept being verbalizable several layers early.
- **`sky-color`** — "…the color of the sky is". Layer 20 surfaces `' blue'`,
  `' rainbow'`, `' colored'`, `' color'`; output is `' blue'`.
- **`boot-country-currency`** — "The currency of the country shaped like a boot
  is the". Layer 20 surfaces `' currency'`, `' dollar'`, `' euro'` — `' euro'`
  (Italy's currency) is present but ranks below `' dollar'`, so this is a
  *partial* success: the currency frame is recovered, the specific answer is not
  cleanly dominant.

## Ambiguous / partial

- **`multi-hop-author`** — "The author of Romeo and Juliet was born in the
  country of". Layers 8–12 surface country concepts including `' England'` and
  `' Britain'`, but `' France'` and `' Europe'` rank comparably or higher. The
  correct concept appears in the candidate set but the multi-hop is not resolved
  — a good illustration of "top-ranked tokens may be misleading."
- **`code-bug`** — a function that uses `-` instead of `+`. Layer 20 surfaces
  `' syntax'`, `' parameter'`, `' variable'`; output is `' operator'`. Related
  to the bug's domain but not the specific concept (`minus`/`subtraction`).

## Failures (worth showing)

- **`spider-legs`** — "The animal that spins webs has this many legs:". No
  `eight`/`8`/`spider` concept surfaces at any layer; readouts are dominated by
  punctuation (`':'`, `'!:'`) and filler-underscore tokens. A clear miss.
- **`arithmetic-intermediate`** — "Seven multiplied by eight equals". No numeric
  concept (`56`, `fifty`) appears; readouts are dominated by quote/punctuation
  tokens. Arithmetic intermediates are not recovered by this small lens.
- **`opposite-hot`** — "The opposite of hot is". `' cold'` appears only in the
  model *output* row, not in the intermediate J-lens readouts, which are
  dominated by underscore-filler tokens at layers 16–20.

## A systematic artifact to note

At layers 16 and 20 several prompts show top concepts dominated by
underscore-run tokens (`'____'`, `' ______'`). This is characteristic of a
small, under-fit corpus-averaged lens rather than a property of the model, and
it is exactly the kind of redundant/misleading top-ranked token discussed in
`docs/LIMITATIONS.md`. A lens fit on more prompts reduces (but does not
eliminate) it. This is why the UI defaults to showing rankings, offers
whitespace/punctuation filters, and never claims calibrated probabilities.
