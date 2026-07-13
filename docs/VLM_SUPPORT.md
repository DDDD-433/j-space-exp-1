# VLM support

OpenJSpace supports vision-language models by reading out the **language
decoder's** residual stream *after* projected visual embeddings have entered the
sequence. This page documents what that means, what is and isn't supported, and
the SmolVLM adapter specifics.

## What a VLM readout means

The J-lens vocabulary is the **text decoder's vocabulary**. So a readout at an
image-token position answers:

> *Which text concepts is this visual-token activation disposed to influence the
> decoder to verbalize?*

It does **not** decode the raw vision encoder's complete representation, and it
is **not** a segmentation or captioning model. It is the same linear,
corpus-averaged, tokenizer-bound projection used for text, applied to positions
that happen to carry visual information.

## Design choices (v0.1)

1. **Official processor and chat template.** Inputs are built with the model's
   own `AutoProcessor` and chat template; the exact multimodal sequence sent to
   the model is preserved, including tile-delimiter markers and `<image>`
   placeholders.
2. **Language backbone only.** Residual activations are recorded solely from the
   language decoder (`block_output`, the same location as text models). The
   vision encoder and connector are run as part of the forward pass but are not
   themselves lensed — a vision-encoder J-lens would need a justified output
   basis, which is out of scope for the first release.
3. **Position metadata.** Every position is classified as `text`,
   `image_token`, `image_boundary`, `special`, or `unknown`, with image tokens
   carrying `image_index` and (when geometry is available) `patch_index`,
   `patch_row`, `patch_col`. Multiple images are distinguishable by
   `image_index`.
4. **No fabricated geometry.** When the connector merges or resamples patches,
   OpenJSpace does not invent a one-to-one pixel mapping.

## Patch mapping status

Every run reports one of:

| Status | Meaning | UI behavior |
| --- | --- | --- |
| `exact` | one-to-one position → (row, col) patch map exists | clickable patch overlay on the image |
| `approximate` | positions are merged/shuffled patch *groups*; (row, col) is a group location | token strip / grid of groups, labeled approximate |
| `unavailable` | architecture resamples tokens; no spatial map | token strip only, no coordinates |

The status is stored in the run metadata and shown in both the UI and exported
reports.

## SmolVLM / SmolVLM2 (Idefics3) — tested

`HuggingFaceTB/SmolVLM-256M-Instruct` is the lightweight, integration-tested VLM.

- **Architecture:** `Idefics3ForConditionalGeneration`; language backbone is a
  30-layer Llama-style decoder (hidden 576, vocab 49280) at
  `model.text_model`; `lm_head` on the root.
- **Sequence layout:** each image is split into tiles (a crop grid plus a global
  thumbnail). Each tile emits a run of `<image>` placeholder tokens bracketed by
  `<fake_token_around_image>`, `<row_R_col_C>`, and `<global-img>` markers,
  classified as `image_boundary`.
- **Connector:** a pixel-shuffle connector groups a `scale_factor²` (= 16) block
  of vision patches into a single decoder token. With `image_size` 512 and
  `patch_size` 16, a tile's 32×32 patches become an **8×8** grid of 64 decoder
  tokens. Patch mapping is therefore **`approximate`**: each image token
  corresponds to a 4×4 group of vision patches, laid out on the 8×8 grid in
  reading order. Non-square runs fall back to a 1×N strip with no fabricated
  coordinates.
- **`torchvision` required.** The Idefics3 image processor needs `torchvision`;
  install it before using images (`pip install torchvision`).
- **`do_image_splitting`.** The adapter exposes `do_image_splitting` (default
  `True`). Setting it `False` emits a single 8×8 thumbnail tile (~64 image
  tokens), which keeps a CPU forward pass tractable; the examples and tests use
  this. Pass `--split` to `examples/vlm_demo.py` for full tiling.

### Text-only forward

The SmolVLM decoder also runs text-only (no images), which is used to fit a
text lens compatible with the same model. In that mode patch mapping is
`unavailable` and all positions are `text`/`special`.

## Not yet supported

- **Qwen2.5-VL / Qwen3-VL** — planned as the richer adapter; not implemented.
- **Exact patch geometry** for any connector that resamples — reported as
  `approximate`/`unavailable` rather than faked.
- **Vision-encoder J-lenses** — would require a separate, justified output basis.

## Tests

- `tests/test_vlm_mapping.py` — fast, weightless unit tests for modality
  classification, multi-image separation, square vs. strip geometry, and the
  "no fabricated coordinates" guarantee.
- `tests/test_integration_smolvlm.py` (marked `integration`) — real
  SmolVLM-256M: routing, image-position classification, multi-image handling,
  text-only forward, and a full image inspection that round-trips through JSON.
