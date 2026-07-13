"""Local FastAPI application: model loading, lens fitting jobs, inspection runs.

Local-first: artifacts and runs live on the local filesystem under
``OPENJSPACE_HOME``; no database, no telemetry, no external services. All
client-supplied names are validated before touching the filesystem.
"""

from __future__ import annotations

import json
import logging
import re
import threading
import uuid
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from openjspace.config import artifacts_dir, runs_dir, uploads_dir
from openjspace.core.serialization import METADATA_FILENAME, LensCompatibilityError
from openjspace.models.registry import LoadedModel, family_for_architecture
from openjspace.server.jobs import Job, JobCancelled, JobManager
from openjspace.server.schemas import (
    ArtifactInfo,
    DecomposeRequest,
    FitRequest,
    InspectRequest,
    JobInfo,
    LoadModelRequest,
    MergeRequest,
    ModelInfo,
    RunInfo,
    UploadInfo,
)

logger = logging.getLogger(__name__)

_SAFE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
_MAX_UPLOAD_BYTES = 20 * 1024 * 1024
_ALLOWED_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".gif"}


def _validate_name(name: str, what: str) -> str:
    """Reject path traversal and unsafe characters in client-supplied names."""
    if not _SAFE_NAME.match(name) or ".." in name:
        raise HTTPException(400, f"invalid {what} name {name!r}")
    return name


def _resolve_under(base: Path, name: str, what: str) -> Path:
    _validate_name(name, what)
    resolved = (base / name).resolve()
    if not str(resolved).startswith(str(base.resolve())):
        raise HTTPException(400, f"{what} path escapes storage directory")
    return resolved


class ModelCache:
    """Keeps at most one loaded model in memory (they are large)."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._key: tuple[str, str, str] | None = None
        self._loaded: LoadedModel | None = None

    def get(self, model_id: str, device: str, dtype: str) -> LoadedModel:
        from openjspace.models.registry import load_model

        key = (model_id, device, dtype)
        with self._lock:
            if self._loaded is not None and self._key == key:
                return self._loaded
            self._loaded = None  # release before loading the next model
            import gc

            gc.collect()
            loaded = load_model(model_id, device=device, dtype=dtype)
            self._key, self._loaded = key, loaded
            return loaded


def create_app() -> FastAPI:
    app = FastAPI(title="OpenJSpace", version="0.1.0", docs_url="/api/docs")
    jobs = JobManager()
    models = ModelCache()
    for directory in (artifacts_dir(), runs_dir(), uploads_dir()):
        directory.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ #
    # Models
    # ------------------------------------------------------------------ #

    @app.post("/api/models/load", response_model=ModelInfo)
    def load_model_endpoint(request: LoadModelRequest) -> ModelInfo:
        try:
            loaded = models.get(request.model_id, request.device, request.dtype)
        except (ValueError, OSError) as exc:
            raise HTTPException(400, str(exc)) from exc
        adapter = loaded.adapter
        family = family_for_architecture(adapter.architecture)
        return ModelInfo(
            model_id=adapter.model_id,
            architecture=adapter.architecture,
            kind=loaded.kind,
            status=family.status if family else "unsupported",
            n_layers=adapter.n_layers,
            hidden_size=adapter.hidden_size,
            vocab_size=adapter.vocab_size,
            residual_location=adapter.residual_location,
            tokenizer_id=adapter.tokenizer_id,
            device=loaded.device,
            dtype=loaded.dtype,
        )

    @app.get("/api/models/families")
    def model_families() -> list[dict[str, Any]]:
        from openjspace.models.registry import MODEL_FAMILIES

        return [
            {
                "name": f.name,
                "kind": f.kind,
                "status": f.status,
                "example_id": f.example_id,
                "notes": f.notes,
            }
            for f in MODEL_FAMILIES
        ]

    # ------------------------------------------------------------------ #
    # Artifacts
    # ------------------------------------------------------------------ #

    @app.get("/api/artifacts", response_model=list[ArtifactInfo])
    def list_artifacts() -> list[ArtifactInfo]:
        results: list[ArtifactInfo] = []
        for child in sorted(artifacts_dir().iterdir()) if artifacts_dir().is_dir() else []:
            meta_path = child / METADATA_FILENAME
            if child.is_dir() and meta_path.is_file():
                try:
                    metadata = json.loads(meta_path.read_text())
                except (OSError, json.JSONDecodeError):
                    continue
                results.append(ArtifactInfo(name=child.name, path=str(child), metadata=metadata))
        return results

    # ------------------------------------------------------------------ #
    # Fitting and merging (jobs)
    # ------------------------------------------------------------------ #

    @app.post("/api/lenses/fit", response_model=JobInfo)
    def fit_lens(request: FitRequest) -> JobInfo:
        output = _resolve_under(artifacts_dir(), request.artifact_name, "artifact")

        def run(job: Job) -> dict[str, Any]:
            from openjspace.core import fitting
            from openjspace.data.datasets import load_fitting_prompts

            job.log(f"loading dataset {request.dataset!r}")
            prompts = load_fitting_prompts(
                request.dataset, num_prompts=request.num_prompts, seed=request.seed
            )
            if not prompts:
                raise ValueError(f"no prompts loaded from dataset {request.dataset!r}")
            job.total = len(prompts)
            job.log(f"loading model {request.model_id}")
            loaded = models.get(request.model_id, request.device, request.dtype)
            output.mkdir(parents=True, exist_ok=True)
            checkpoint = output / "fit_checkpoint"
            job.checkpoint_path = str(checkpoint)

            def progress(done: int, total: int, message: str) -> None:
                job.progress, job.total = done, total
                job.log(message)

            def should_cancel() -> bool:
                return job.cancel_event.is_set()

            try:
                lens = fitting.fit(
                    loaded.adapter,
                    prompts,
                    source_layers=request.layers,
                    target_layer=request.target_layer,
                    dim_batch=request.dim_batch,
                    max_seq_len=request.max_seq_len,
                    skip_first=request.skip_first,
                    checkpoint_path=checkpoint,
                    checkpoint_every=request.checkpoint_every,
                    fitting_dataset=f"{request.dataset} (num_prompts={request.num_prompts}, "
                    f"seed={request.seed})",
                    model_dtype=loaded.dtype,
                    progress=progress,
                    should_cancel=should_cancel,
                )
            except fitting.FittingCancelled as exc:
                raise JobCancelled(str(exc)) from exc
            lens.save(output)
            fitting.cleanup_checkpoint(checkpoint)
            job.log(f"saved artifact to {output}")
            return {"artifact_name": request.artifact_name, "path": str(output)}

        job = jobs.submit("fit", run, total=request.num_prompts)
        return JobInfo(**job.to_dict())

    @app.post("/api/lenses/merge", response_model=JobInfo)
    def merge_lenses(request: MergeRequest) -> JobInfo:
        shard_paths = [
            _resolve_under(artifacts_dir(), name, "artifact") for name in request.shard_names
        ]
        output = _resolve_under(artifacts_dir(), request.output_name, "artifact")

        def run(job: Job) -> dict[str, Any]:
            from openjspace.core.lens import JacobianLens

            job.total = len(shard_paths)
            lenses = []
            for i, path in enumerate(shard_paths):
                lenses.append(JacobianLens.load(path))
                job.progress = i + 1
            merged = JacobianLens.merge(lenses)
            merged.save(output)
            return {"artifact_name": request.output_name, "path": str(output)}

        job = jobs.submit("merge", run, total=len(shard_paths))
        return JobInfo(**job.to_dict())

    # ------------------------------------------------------------------ #
    # Jobs
    # ------------------------------------------------------------------ #

    @app.get("/api/jobs/{job_id}", response_model=JobInfo)
    def get_job(job_id: str) -> JobInfo:
        job = jobs.get(_validate_name(job_id, "job"))
        if job is None:
            raise HTTPException(404, f"job {job_id} not found")
        return JobInfo(**job.to_dict())

    @app.post("/api/jobs/{job_id}/cancel", response_model=JobInfo)
    def cancel_job(job_id: str) -> JobInfo:
        job = jobs.cancel(_validate_name(job_id, "job"))
        if job is None:
            raise HTTPException(404, f"job {job_id} not found")
        return JobInfo(**job.to_dict())

    # ------------------------------------------------------------------ #
    # Uploads
    # ------------------------------------------------------------------ #

    @app.post("/api/uploads", response_model=UploadInfo)
    async def upload_image(file: UploadFile) -> UploadInfo:
        from PIL import Image as PILImage

        suffix = Path(file.filename or "upload.png").suffix.lower()
        if suffix not in _ALLOWED_IMAGE_SUFFIXES:
            raise HTTPException(
                400, f"unsupported file type {suffix!r}; allowed: {sorted(_ALLOWED_IMAGE_SUFFIXES)}"
            )
        data = await file.read()
        if len(data) > _MAX_UPLOAD_BYTES:
            raise HTTPException(400, f"file too large (> {_MAX_UPLOAD_BYTES} bytes)")
        name = f"{uuid.uuid4().hex[:12]}{suffix}"
        path = uploads_dir() / name
        path.write_bytes(data)
        try:
            with PILImage.open(path) as image:
                image.verify()
            with PILImage.open(path) as image:
                width, height = image.size
        except Exception as exc:
            path.unlink(missing_ok=True)
            raise HTTPException(400, f"not a valid image: {exc}") from exc
        return UploadInfo(name=name, width=width, height=height)

    @app.get("/api/uploads/{name}")
    def get_upload(name: str) -> FileResponse:
        path = _resolve_under(uploads_dir(), name, "upload")
        if not path.is_file():
            raise HTTPException(404, "upload not found")
        return FileResponse(path)

    # ------------------------------------------------------------------ #
    # Inspection and decomposition
    # ------------------------------------------------------------------ #

    def _load_images(names: list[str]) -> list[Any] | None:
        if not names:
            return None
        from PIL import Image as PILImage

        images = []
        for name in names:
            path = _resolve_under(uploads_dir(), name, "upload")
            if not path.is_file():
                raise HTTPException(400, f"upload {name!r} not found")
            images.append(PILImage.open(path).convert("RGB"))
        return images

    @app.post("/api/inspect")
    def inspect(request: InspectRequest) -> JSONResponse:
        from openjspace.core.applying import inspect_prompt
        from openjspace.core.lens import JacobianLens

        lens_path = _resolve_under(artifacts_dir(), request.lens_name, "artifact")
        if not (lens_path / METADATA_FILENAME).is_file():
            raise HTTPException(400, f"lens artifact {request.lens_name!r} not found")
        try:
            loaded = models.get(request.model_id, request.device, request.dtype)
            lens = JacobianLens.load(lens_path)
        except (ValueError, OSError) as exc:
            raise HTTPException(400, str(exc)) from exc

        tracked_ids = list(request.tracked_token_ids)
        tokenizer = getattr(loaded.adapter, "tokenizer", None)
        if request.tracked_concepts and tokenizer is not None:
            for concept in request.tracked_concepts:
                ids = tokenizer.encode(concept, add_special_tokens=False)
                if ids:
                    tracked_ids.append(int(ids[0]))

        try:
            result = inspect_prompt(
                loaded.adapter,
                lens,
                request.prompt,
                images=_load_images(request.image_names),
                layers=request.layers,
                positions=request.positions,
                top_k=request.top_k,
                max_seq_len=request.max_seq_len,
                use_chat_template=request.use_chat_template,
                tracked_token_ids=tracked_ids,
                force=request.force,
                lens_path=str(lens_path),
                device=loaded.device,
                dtype=loaded.dtype,
                model_kind=loaded.kind,
            )
        except LensCompatibilityError as exc:
            raise HTTPException(409, str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc

        run_dir = runs_dir() / result.metadata.run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "run.json").write_text(result.model_dump_json(), encoding="utf-8")
        return JSONResponse(json.loads(result.model_dump_json()))

    @app.post("/api/decompose")
    def decompose(request: DecomposeRequest) -> JSONResponse:
        from openjspace.analysis.decompose_runner import decompose_cell
        from openjspace.core.lens import JacobianLens

        lens_path = _resolve_under(artifacts_dir(), request.lens_name, "artifact")
        if not (lens_path / METADATA_FILENAME).is_file():
            raise HTTPException(400, f"lens artifact {request.lens_name!r} not found")
        try:
            loaded = models.get(request.model_id, request.device, request.dtype)
            lens = JacobianLens.load(lens_path)
            record = decompose_cell(
                loaded.adapter,
                lens,
                request.prompt,
                images=_load_images(request.image_names),
                layer=request.layer,
                position=request.position,
                k=request.k,
                use_chat_template=request.use_chat_template,
                max_seq_len=request.max_seq_len,
            )
        except (ValueError, OSError) as exc:
            raise HTTPException(400, str(exc)) from exc
        return JSONResponse(json.loads(record.model_dump_json()))

    # ------------------------------------------------------------------ #
    # Runs
    # ------------------------------------------------------------------ #

    @app.get("/api/runs", response_model=list[RunInfo])
    def list_runs() -> list[RunInfo]:
        results: list[RunInfo] = []
        for child in sorted(runs_dir().iterdir()) if runs_dir().is_dir() else []:
            run_path = child / "run.json"
            if run_path.is_file():
                try:
                    data = json.loads(run_path.read_text())
                    meta = data.get("metadata", {})
                    results.append(
                        RunInfo(
                            run_id=meta.get("run_id", child.name),
                            created_at=meta.get("created_at", ""),
                            model_id=meta.get("model_id", ""),
                            prompt=meta.get("prompt", "")[:200],
                        )
                    )
                except (OSError, json.JSONDecodeError):
                    continue
        return results

    @app.get("/api/runs/{run_id}")
    def get_run(run_id: str) -> JSONResponse:
        run_path = _resolve_under(runs_dir(), run_id, "run") / "run.json"
        if not run_path.is_file():
            raise HTTPException(404, f"run {run_id} not found")
        return JSONResponse(json.loads(run_path.read_text()))

    @app.get("/api/runs/{run_id}/report.html")
    def get_run_report(run_id: str) -> HTMLResponse:
        from openjspace.report.html_export import render_html
        from openjspace.report.schema import RunResult

        run_path = _resolve_under(runs_dir(), run_id, "run") / "run.json"
        if not run_path.is_file():
            raise HTTPException(404, f"run {run_id} not found")
        result = RunResult.model_validate_json(run_path.read_text())
        return HTMLResponse(render_html(result))

    # ------------------------------------------------------------------ #
    # Static frontend (built React app), when present
    # ------------------------------------------------------------------ #

    web_dist = Path(__file__).resolve().parents[3] / "web" / "dist"
    if web_dist.is_dir():
        app.mount("/", StaticFiles(directory=web_dist, html=True), name="web")
    else:

        @app.get("/", response_class=HTMLResponse)
        def index() -> str:
            return (
                "<h1>OpenJSpace API</h1><p>The web UI is not built. Run "
                "<code>cd web && npm install && npm run build</code>, then restart. "
                'API docs at <a href="/api/docs">/api/docs</a>.</p>'
            )

    return app
