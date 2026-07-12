#!/usr/bin/env python
"""VLM demo: run SmolVLM on an image + prompt and read out the language-decoder
residual stream at image-token and text-token positions.

The J-lens vocabulary is the text decoder's vocabulary, so an image-position
readout answers *"which text concepts is this visual-token activation disposed
to influence the decoder to verbalize?"* — it is NOT a decode of the vision
encoder's full representation. Patch mapping for SmolVLM's pixel-shuffle
connector is APPROXIMATE (each decoder token is a group of vision patches).

Because a full multi-tile image forward is slow on CPU, this demo disables
image splitting by default (one 8x8 thumbnail tile, ~64 image tokens).

Usage::

    python examples/vlm_demo.py --image path/to/photo.jpg --prompt "What is this?"
    python examples/vlm_demo.py   # generates a synthetic test image

A tiny text-only lens is fit on the fly (the language decoder still runs
text-only), purely so there is a compatible artifact to demonstrate readout;
it is not a high-quality lens.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from openjspace.core.applying import inspect_prompt
from openjspace.core.fitting import fit
from openjspace.core.lens import JacobianLens
from openjspace.logging_utils import configure_logging
from openjspace.models.registry import load_model
from openjspace.report.html_export import export_html

HERE = Path(__file__).parent
DEFAULT_MODEL = "HuggingFaceTB/SmolVLM-256M-Instruct"
DEFAULT_LENS = HERE.parent / "artifacts" / "smolvlm-256m-demo"
DEMO_LAYER = 12

FIT_TEXT = [
    "A photograph usually shows objects, people, animals, or scenery arranged "
    "within a rectangular frame, lit by natural or artificial light and framed "
    "from a particular point of view.",
    "Describing a picture involves naming the main subject, the background, the "
    "colors, and the spatial relationships between the things that appear in it.",
]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--lens", type=Path, default=DEFAULT_LENS)
    parser.add_argument("--image", type=Path, default=None, help="Image file (else synthetic)")
    parser.add_argument("--prompt", default="What is in this image?")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--layer", type=int, default=DEMO_LAYER)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--split", action="store_true", help="Enable full image tiling (slow)")
    parser.add_argument("--out", type=Path, default=HERE.parent / "runs" / "vlm_demo")
    return parser


def load_image(path: Path | None):
    from PIL import Image

    if path is not None:
        return Image.open(path).convert("RGB")
    import numpy as np

    rng = np.random.default_rng(0)
    return Image.fromarray((rng.random((64, 64, 3)) * 255).astype("uint8"))


def main() -> None:
    configure_logging()
    args = build_parser().parse_args()

    loaded = load_model(args.model, device=args.device, dtype=args.dtype)
    adapter = loaded.adapter
    if hasattr(adapter, "do_image_splitting"):
        adapter.do_image_splitting = args.split

    if (args.lens / "metadata.json").is_file():
        lens = JacobianLens.load(args.lens)
    else:
        print(f"Fitting tiny text-only demo lens -> {args.lens}")
        lens = fit(
            adapter,
            FIT_TEXT,
            source_layers=[args.layer],
            dim_batch=64,
            max_seq_len=48,
            skip_first=4,
            fitting_dataset="examples/vlm_demo.py inline text",
            model_dtype=loaded.dtype,
        )
        lens.save(args.lens)

    image = load_image(args.image)
    inputs = adapter.prepare_inputs(args.prompt, images=[image], max_length=8192)
    result = inspect_prompt(
        adapter,
        lens,
        args.prompt,
        prepared_inputs=inputs,
        layers=[args.layer],
        positions="all",
        top_k=args.top_k,
        model_kind="vlm",
        lens_path=str(args.lens),
        device=loaded.device,
        dtype=loaded.dtype,
    )

    print(f"\npatch mapping: {result.metadata.patch_mapping}")
    for warning in result.metadata.warnings:
        print(f"warning: {warning}")

    image_positions = [p for p in result.positions if p.modality == "image_token"]
    text_positions = [p for p in result.positions if p.modality == "text"]
    print(f"\n{len(image_positions)} image-token positions, {len(text_positions)} text positions")

    def show(label: str, index: int) -> None:
        cell = result.cell(args.layer, index)
        if cell is None:
            return
        top = ", ".join(f"{c.token_display!r}" for c in cell.jlens_top[:args.top_k])
        print(f"  {label} (pos {index}): {top}")

    print(f"\nTop J-lens concepts at layer {args.layer}:")
    for meta in image_positions[:4]:
        rc = f"[r{meta.patch_row},c{meta.patch_col}]" if meta.patch_row is not None else ""
        show(f"image token {rc}", meta.index)
    for meta in text_positions[-3:]:
        show(f"text {meta.token_text!r}", meta.index)

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "run.json").write_text(result.model_dump_json(indent=2), encoding="utf-8")
    export_html(result, args.out / "report.html")
    print(f"\nwrote {args.out}/report.html")


if __name__ == "__main__":
    main()
