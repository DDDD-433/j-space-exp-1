# OpenJSpace examples

Runnable demonstrations of the Jacobian-lens pipeline. **These are
demonstrations, not benchmarks.** The bundled lenses are fit on a handful of
short prompts so they run on a laptop CPU in minutes; readouts are approximate,
corpus-averaged concept readouts and should be read as hypotheses to inspect,
not ground truth. See `docs/LIMITATIONS.md`.

## Files

| File | What it does |
| --- | --- |
| `sample_prompts.jsonl` | Small fitting corpus (8 encyclopedic paragraphs). |
| `eval_examples.jsonl` | Qualitative evaluation prompts (implicit entity, geography, multi-hop, arithmetic, code bug, commonsense). |
| `text_demo.py` | Fit a small Qwen2.5-0.5B lens (if needed), then read out the eval prompts and write JSON + HTML reports. |
| `vlm_demo.py` | Run SmolVLM on an image + prompt and read out the language-decoder residual stream at image-token and text-token positions. |
| `reports/` | Committed self-contained HTML reports from example runs. |

## Text demo

```bash
# Fit a demo lens (a few minutes on CPU) and inspect all eval prompts:
python examples/text_demo.py --device cpu --dtype float32

# Reuse an already-fit lens and run one example:
python examples/text_demo.py --skip-fit --only spider-legs
```

Reports are written under `runs/text_demo/<slug>/report.html`.

## VLM demo

Requires `torchvision` (the SmolVLM/Idefics3 image processor needs it):

```bash
pip install torchvision
python examples/vlm_demo.py --device cpu --dtype float32   # uses a synthetic image
python examples/vlm_demo.py --image path/to/photo.jpg --prompt "What is in this image?"
```

Image splitting is disabled by default so a single 8×8 thumbnail tile
(~64 image tokens) keeps the CPU forward tractable; pass `--split` for the full
tiling the model normally uses. Patch mapping is **approximate** for SmolVLM's
pixel-shuffle connector — each decoder image token corresponds to a *group* of
vision patches, and the reports label it as such.

See `reports/vlm_smolvlm_synthetic/report.html` for a committed example run on a
synthetic random-noise image (the image-token readouts surface color / spectrum
concepts, which is the kind of qualitative, non-cherry-picked result to expect).
