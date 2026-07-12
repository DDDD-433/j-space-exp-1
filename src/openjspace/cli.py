"""OpenJSpace command-line interface."""

from __future__ import annotations

import platform
import sys
from pathlib import Path
from typing import Annotated

import typer

from openjspace.logging_utils import configure_logging

app = typer.Typer(
    name="openjspace",
    help=(
        "Interactive Jacobian-lens and J-space visualizer for open-weight "
        "language models and VLMs. Readouts are approximate, corpus-averaged "
        "concept readouts — not literal model 'thoughts'."
    ),
    no_args_is_help=True,
    pretty_exceptions_show_locals=False,
)

models_app = typer.Typer(help="List and inspect supported models.", no_args_is_help=True)
model_app = typer.Typer(help="Inspect a specific model's structure.", no_args_is_help=True)
app.add_typer(models_app, name="models")
app.add_typer(model_app, name="model")


def _parse_layers(layers: str | None) -> list[int] | None:
    if layers is None or layers.strip() in ("", "all"):
        return None
    return [int(part) for part in layers.split(",") if part.strip()]


@app.command()
def doctor() -> None:
    """Report environment, devices, and likely fitting compatibility."""
    import torch

    import openjspace
    from openjspace.config import device_info, resolve_device

    typer.echo(f"openjspace     {openjspace.__version__}")
    typer.echo(f"python         {platform.python_version()} ({sys.executable})")
    typer.echo(f"pytorch        {torch.__version__}")
    try:
        import transformers

        typer.echo(f"transformers   {transformers.__version__}")
    except ImportError:
        typer.echo("transformers   NOT INSTALLED")
    try:
        import safetensors

        typer.echo(f"safetensors    {safetensors.__version__}")
    except ImportError:
        typer.echo("safetensors    NOT INSTALLED")

    devices = ["cpu"]
    if torch.cuda.is_available():
        devices.insert(0, f"cuda ({torch.cuda.get_device_name(0)})")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        devices.insert(0, "mps")
    typer.echo(f"devices        {', '.join(devices)}")
    selected = resolve_device("auto")
    info = device_info(selected)
    typer.echo(f"selected       {selected}")
    typer.echo(f"bf16 support   {info.supports_bf16}")
    typer.echo(f"fp16 support   {info.supports_fp16}")

    try:
        import psutil  # type: ignore[import-not-found]

        mem_gb = psutil.virtual_memory().total / 1e9
        typer.echo(f"system memory  {mem_gb:.1f} GB")
    except ImportError:
        try:
            import os

            pages = os.sysconf("SC_PHYS_PAGES")
            page_size = os.sysconf("SC_PAGE_SIZE")
            typer.echo(f"system memory  {pages * page_size / 1e9:.1f} GB")
        except (ValueError, OSError):
            typer.echo("system memory  not detectable")

    typer.echo("")
    if selected == "cpu":
        typer.echo(
            "fitting        CPU-only: fine for tiny/small models (<= ~0.5B); "
            "expect slow fits for larger ones."
        )
    else:
        typer.echo(f"fitting        {selected} available; use --dim-batch to control memory.")
    typer.echo(
        "note           quantized checkpoints are rejected for fitting "
        "(experimental future work); lens accumulation is always float32."
    )


@models_app.command("list")
def models_list() -> None:
    """Show the supported-model table with statuses."""
    from openjspace.models.registry import MODEL_FAMILIES

    width = max(len(f.name) for f in MODEL_FAMILIES) + 2
    typer.echo(f"{'FAMILY':<{width}}{'KIND':<6}{'STATUS':<14}EXAMPLE")
    for family in MODEL_FAMILIES:
        typer.echo(f"{family.name:<{width}}{family.kind:<6}{family.status:<14}{family.example_id}")
        if family.notes:
            typer.echo(f"{'':<{width}}{'':<6}{'':<14}({family.notes})")
    typer.echo(
        "\n'tested' means an integration test has passed; 'experimental' means "
        "the layout is supported but untested against real weights."
    )


@model_app.command("inspect")
def model_inspect(
    model: Annotated[str, typer.Option(help="HF model id or local path")],
    device: str = "cpu",
    dtype: str = "auto",
    revision: str | None = None,
) -> None:
    """Load a model and print its lens-relevant structure."""
    configure_logging()
    from openjspace.models.registry import family_for_architecture, load_model

    loaded = load_model(model, device=device, dtype=dtype, revision=revision)
    adapter = loaded.adapter
    family = family_for_architecture(adapter.architecture)
    typer.echo(f"model_id           {adapter.model_id}")
    typer.echo(f"architecture       {adapter.architecture}")
    typer.echo(f"kind               {loaded.kind}")
    typer.echo(f"status             {family.status if family else 'unsupported/unknown'}")
    typer.echo(f"n_layers           {adapter.n_layers}")
    typer.echo(f"hidden_size        {adapter.hidden_size}")
    typer.echo(f"vocab_size         {adapter.vocab_size}")
    typer.echo(f"residual_location  {adapter.residual_location}")
    typer.echo(f"tokenizer_id       {adapter.tokenizer_id}")
    typer.echo(f"device             {loaded.device}")
    typer.echo(f"dtype              {loaded.dtype}")


@app.command()
def fit(
    model: Annotated[str, typer.Option(help="HF model id or local path")],
    dataset: Annotated[str, typer.Option(help="'wikitext', a .jsonl file, or a .txt file")],
    output: Annotated[Path, typer.Option(help="Output artifact directory")],
    layers: Annotated[
        str | None, typer.Option(help="Comma-separated source layers (default: all below target)")
    ] = None,
    target_layer: Annotated[int | None, typer.Option(help="Target layer (default: final)")] = None,
    max_seq_len: int = 128,
    num_prompts: int = 100,
    dim_batch: int = 8,
    skip_first: int = 16,
    device: str = "auto",
    dtype: str = "auto",
    seed: int = 0,
    shard_index: int = 0,
    num_shards: int = 1,
    checkpoint_every: int = 10,
    resume: bool = True,
    revision: str | None = None,
) -> None:
    """Fit a Jacobian lens on a model over a prompt corpus.

    One forward pass per prompt, ceil(hidden/dim_batch) backward passes;
    accumulation in float32; checkpoints are atomic and resumable. Shard with
    --shard-index/--num-shards and combine with `openjspace merge`.
    """
    configure_logging()
    from openjspace.core import fitting
    from openjspace.data.datasets import load_fitting_prompts
    from openjspace.models.registry import load_model

    prompts = load_fitting_prompts(
        dataset, num_prompts=num_prompts, seed=seed, shard_index=shard_index, num_shards=num_shards
    )
    if not prompts:
        typer.echo("error: no prompts loaded from dataset", err=True)
        raise typer.Exit(1)
    typer.echo(f"loaded {len(prompts)} prompts (shard {shard_index}/{num_shards}, seed {seed})")

    loaded = load_model(model, device=device, dtype=dtype, revision=revision)
    output.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output / "fit_checkpoint"

    def progress(done: int, total: int, message: str) -> None:
        typer.echo(f"  [{done}/{total}] {message}")

    lens = fitting.fit(
        loaded.adapter,
        prompts,
        source_layers=_parse_layers(layers),
        target_layer=target_layer,
        dim_batch=dim_batch,
        max_seq_len=max_seq_len,
        skip_first=skip_first,
        checkpoint_path=checkpoint_path,
        checkpoint_every=checkpoint_every,
        resume=resume,
        fitting_dataset=f"{dataset} (num_prompts={num_prompts}, seed={seed}, "
        f"shard={shard_index}/{num_shards})",
        model_dtype=loaded.dtype,
        progress=progress,
    )
    lens.save(output)
    fitting.cleanup_checkpoint(checkpoint_path)
    typer.echo(f"saved lens artifact to {output}")


@app.command()
def merge(
    shards: Annotated[list[Path], typer.Argument(help="Shard artifact directories")],
    output: Annotated[Path, typer.Option(help="Merged artifact directory")] = Path("merged"),
) -> None:
    """Merge lens shards, weighting by each shard's accumulated prompt count.

    Shards must agree on model identity, layers, residual location, and
    fitting hyperparameters; validation runs before any merge.
    """
    configure_logging()
    from openjspace.core.lens import JacobianLens

    lenses = [JacobianLens.load(shard) for shard in shards]
    merged = JacobianLens.merge(lenses)
    merged.save(output)
    typer.echo(
        f"merged {len(lenses)} shards "
        f"({merged.metadata.number_of_prompts} prompts total) into {output}"
    )


@app.command()
def inspect(
    model: Annotated[str, typer.Option(help="HF model id or local path")],
    lens: Annotated[Path, typer.Option(help="Lens artifact directory")],
    prompt: Annotated[str, typer.Option(help="Prompt text")],
    output: Annotated[Path, typer.Option(help="Run output directory")] = Path("runs/latest"),
    image: Annotated[list[Path] | None, typer.Option(help="Image file(s) for VLMs")] = None,
    layers: str | None = None,
    positions: str = "all",
    top_k: int = 10,
    max_seq_len: int = 512,
    chat: Annotated[bool, typer.Option(help="Apply the tokenizer chat template")] = False,
    track: Annotated[
        str | None, typer.Option(help="Comma-separated concept strings to pin/track")
    ] = None,
    device: str = "auto",
    dtype: str = "auto",
    force: Annotated[bool, typer.Option(help="Apply an incompatible lens (warned)")] = False,
    html: Annotated[bool, typer.Option(help="Also write a self-contained HTML report")] = True,
) -> None:
    """Run a prompt, apply the lens across layers x positions, save run JSON."""
    configure_logging()
    from openjspace.core.applying import inspect_prompt
    from openjspace.core.lens import JacobianLens
    from openjspace.models.registry import load_model
    from openjspace.report.html_export import export_html

    loaded = load_model(model, device=device, dtype=dtype)
    lens_obj = JacobianLens.load(lens)

    images: list[object] = []
    if image:
        from PIL import Image as PILImage

        images = [PILImage.open(p).convert("RGB") for p in image]

    tracked_ids: list[int] = []
    if track:
        tokenizer = getattr(loaded.adapter, "tokenizer", None)
        if tokenizer is None:
            typer.echo("warning: adapter has no tokenizer; --track ignored", err=True)
        else:
            for concept in track.split(","):
                ids = tokenizer.encode(concept, add_special_tokens=False)
                if len(ids) != 1:
                    typer.echo(
                        f"warning: {concept!r} tokenizes to {len(ids)} tokens; "
                        "tracking its first token only (multi-token concepts are "
                        "poorly represented in a single-token lens)",
                        err=True,
                    )
                tracked_ids.append(ids[0])

    result = inspect_prompt(
        loaded.adapter,
        lens_obj,
        prompt,
        images=images or None,
        layers=_parse_layers(layers),
        positions=positions,
        top_k=top_k,
        max_seq_len=max_seq_len,
        use_chat_template=chat,
        tracked_token_ids=tracked_ids,
        force=force,
        lens_path=str(lens),
        device=loaded.device,
        dtype=loaded.dtype,
        model_kind=loaded.kind,
    )
    for warning in result.metadata.warnings:
        typer.echo(f"warning: {warning}", err=True)

    output.mkdir(parents=True, exist_ok=True)
    run_path = output / "run.json"
    run_path.write_text(result.model_dump_json(indent=2), encoding="utf-8")
    typer.echo(f"wrote {run_path}")
    if html:
        html_path = output / "report.html"
        export_html(result, html_path)
        typer.echo(f"wrote {html_path}")


@app.command()
def decompose(
    model: Annotated[str, typer.Option(help="HF model id or local path")],
    lens: Annotated[Path, typer.Option(help="Lens artifact directory")],
    prompt: Annotated[str, typer.Option(help="Prompt text")],
    layer: Annotated[int, typer.Option(help="Source layer to decompose at")],
    position: Annotated[int, typer.Option(help="Token position (negative from end)")] = -1,
    k: Annotated[int, typer.Option(help="Sparsity budget (max atoms)")] = 10,
    device: str = "auto",
    dtype: str = "auto",
    chat: bool = False,
    output: Annotated[Path | None, typer.Option(help="Write JSON result here")] = None,
) -> None:
    """Sparse non-negative J-space decomposition of one activation.

    Reports selected concepts, coefficients, reconstruction error, and
    explained norm. The decomposition is non-unique (correlated, overcomplete
    dictionary).
    """
    configure_logging()
    from openjspace.analysis.decompose_runner import decompose_cell
    from openjspace.core.lens import JacobianLens
    from openjspace.models.registry import load_model

    loaded = load_model(model, device=device, dtype=dtype)
    lens_obj = JacobianLens.load(lens)
    record = decompose_cell(
        loaded.adapter,
        lens_obj,
        prompt,
        layer=layer,
        position=position,
        k=k,
        use_chat_template=chat,
    )
    typer.echo(f"layer {record.layer}, position {record.position}, k<={record.k_requested}")
    for entry in record.entries:
        typer.echo(f"  {entry.coefficient:10.4f}  {entry.token_display!r} (id {entry.token_id})")
    typer.echo(f"reconstruction_error     {record.reconstruction_error:.4f}")
    typer.echo(f"explained_norm_fraction  {record.explained_norm_fraction:.4f}")
    typer.echo(f"note: {record.warning}")
    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(record.model_dump_json(indent=2), encoding="utf-8")
        typer.echo(f"wrote {output}")


@app.command()
def export(
    run: Annotated[Path, typer.Argument(help="Run directory or run.json path")],
    output: Annotated[Path | None, typer.Option(help="Output file (.html or .json)")] = None,
    format: Annotated[str, typer.Option(help="'html' or 'json'")] = "html",
) -> None:
    """Export a saved run to self-contained HTML or normalized JSON."""
    from openjspace.report.html_export import export_html
    from openjspace.report.schema import RunResult

    run_path = run / "run.json" if run.is_dir() else run
    if not run_path.is_file():
        typer.echo(f"error: {run_path} not found", err=True)
        raise typer.Exit(1)
    result = RunResult.model_validate_json(run_path.read_text(encoding="utf-8"))
    if format == "html":
        out = output or run_path.with_name("report.html")
        export_html(result, out)
    elif format == "json":
        out = output or run_path.with_name("run_export.json")
        out.write_text(result.model_dump_json(indent=2), encoding="utf-8")
    else:
        typer.echo(f"error: unknown format {format!r}", err=True)
        raise typer.Exit(1)
    typer.echo(f"wrote {out}")


@app.command()
def serve(
    host: str = "127.0.0.1",
    port: int = 8000,
    reload: bool = False,
) -> None:
    """Start the local web interface (FastAPI + bundled React app)."""
    configure_logging()
    import uvicorn

    uvicorn.run(
        "openjspace.server.app:create_app", host=host, port=port, reload=reload, factory=True
    )


if __name__ == "__main__":
    app()
