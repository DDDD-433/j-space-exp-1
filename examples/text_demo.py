#!/usr/bin/env python
"""End-to-end text demo: fit a small Jacobian lens on Qwen2.5-0.5B, then run
the qualitative evaluation prompts through it and write JSON + HTML reports.

This is a *demonstration*, not a benchmark. The lens is fit on a handful of
short prompts so it runs on a laptop CPU in a few minutes; readouts are
approximate, corpus-averaged concept readouts and should be read as such.

Usage::

    python examples/text_demo.py --help
    python examples/text_demo.py                 # fit (if needed) + inspect all
    python examples/text_demo.py --skip-fit      # reuse an existing lens
    python examples/text_demo.py --only spider-legs

The default settings mirror ``artifacts/qwen2.5-0.5b-instruct-demo`` produced
by the project's demo fit; see ``examples/RESULTS.md`` for observed outputs.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from openjspace.core.applying import inspect_prompt
from openjspace.core.fitting import cleanup_checkpoint, fit
from openjspace.core.lens import JacobianLens
from openjspace.data.datasets import load_fitting_prompts
from openjspace.data.prompt_loading import load_examples_jsonl
from openjspace.logging_utils import configure_logging
from openjspace.models.registry import load_model
from openjspace.report.html_export import export_html

HERE = Path(__file__).parent
DEFAULT_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
DEFAULT_LENS = HERE.parent / "artifacts" / "qwen2.5-0.5b-instruct-demo"
DEFAULT_LAYERS = [2, 4, 8, 12, 16, 20]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL, help="HF model id")
    parser.add_argument("--lens", type=Path, default=DEFAULT_LENS, help="Lens artifact directory")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", default="auto")
    parser.add_argument(
        "--dataset",
        default=str(HERE / "sample_prompts.jsonl"),
        help="Fitting corpus (.jsonl/.txt or 'wikitext')",
    )
    parser.add_argument("--num-prompts", type=int, default=8)
    parser.add_argument("--max-seq-len", type=int, default=64)
    parser.add_argument("--dim-batch", type=int, default=32)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--skip-fit", action="store_true", help="Reuse an existing lens")
    parser.add_argument("--only", default=None, help="Run just one example slug")
    parser.add_argument(
        "--out", type=Path, default=HERE.parent / "runs" / "text_demo", help="Report output dir"
    )
    return parser


def ensure_lens(args: argparse.Namespace) -> None:
    if args.skip_fit or (args.lens / "metadata.json").is_file():
        return
    print(f"Fitting demo lens -> {args.lens}")
    loaded = load_model(args.model, device=args.device, dtype=args.dtype)
    prompts = load_fitting_prompts(args.dataset, num_prompts=args.num_prompts, seed=0)
    checkpoint = args.lens / "fit_checkpoint"
    lens = fit(
        loaded.adapter,
        prompts,
        source_layers=DEFAULT_LAYERS,
        dim_batch=args.dim_batch,
        max_seq_len=args.max_seq_len,
        checkpoint_path=checkpoint,
        fitting_dataset=f"examples/sample_prompts.jsonl (num_prompts={args.num_prompts})",
        model_dtype=loaded.dtype,
    )
    lens.save(args.lens)
    cleanup_checkpoint(checkpoint)


def main() -> None:
    configure_logging()
    args = build_parser().parse_args()
    ensure_lens(args)

    loaded = load_model(args.model, device=args.device, dtype=args.dtype)
    lens = JacobianLens.load(args.lens)
    examples = load_examples_jsonl(HERE / "eval_examples.jsonl")
    if args.only:
        examples = [e for e in examples if e.slug == args.only]
        if not examples:
            raise SystemExit(f"no example with slug {args.only!r}")

    args.out.mkdir(parents=True, exist_ok=True)
    for example in examples:
        prompt = example.prompt or example.user or ""
        print(f"\n=== {example.slug}: {prompt!r}")
        print(f"    expecting concepts like: {', '.join(example.expected_concepts)}")
        result = inspect_prompt(
            loaded.adapter,
            lens,
            prompt,
            positions="last:1",
            top_k=args.top_k,
            model_kind="text",
            lens_path=str(args.lens),
            device=loaded.device,
            dtype=loaded.dtype,
        )
        run_dir = args.out / example.slug
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "run.json").write_text(result.model_dump_json(indent=2), encoding="utf-8")
        export_html(result, run_dir / "report.html")
        for cell in result.cells:
            if cell.position == result.cells[-1].position:
                tag = "output" if cell.is_model_output else f"L{cell.layer:>2} J-lens"
                top = ", ".join(f"{c.token_display!r}" for c in cell.jlens_top[:5])
                print(f"    {tag}: {top}")
        print(f"    wrote {run_dir}/report.html")


if __name__ == "__main__":
    main()
