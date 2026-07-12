"""In-process job queue for long-running work (lens fitting).

One background worker thread executes jobs sequentially (model forward/backward
passes should not run concurrently in-process). Jobs expose status, progress
counters, capped logs, error details, a checkpoint path, and cooperative
cancellation. This is deliberately not a distributed system.
"""

from __future__ import annotations

import logging
import queue
import threading
import traceback
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Literal

logger = logging.getLogger(__name__)

JobStatus = Literal["queued", "running", "succeeded", "failed", "cancelled"]

MAX_LOG_LINES = 500


@dataclass
class Job:
    job_id: str
    kind: str
    status: JobStatus = "queued"
    progress: int = 0
    total: int = 0
    logs: list[str] = field(default_factory=list)
    error: str | None = None
    checkpoint_path: str | None = None
    result: dict[str, Any] | None = None
    cancel_event: threading.Event = field(default_factory=threading.Event)

    def log(self, message: str) -> None:
        self.logs.append(message)
        if len(self.logs) > MAX_LOG_LINES:
            del self.logs[: len(self.logs) - MAX_LOG_LINES]

    def to_dict(self) -> dict[str, Any]:
        return {
            "job_id": self.job_id,
            "kind": self.kind,
            "status": self.status,
            "progress": self.progress,
            "total": self.total,
            "logs": self.logs[-50:],
            "error": self.error,
            "checkpoint_path": self.checkpoint_path,
            "result": self.result,
        }


JobFunc = Callable[[Job], dict[str, Any] | None]


class JobCancelled(RuntimeError):
    """Raised inside a job function to signal cooperative cancellation."""


class JobManager:
    """Sequential in-process job executor with a single worker thread."""

    def __init__(self) -> None:
        self._jobs: dict[str, Job] = {}
        self._queue: queue.Queue[tuple[Job, JobFunc] | None] = queue.Queue()
        self._lock = threading.Lock()
        self._worker: threading.Thread | None = None

    def _ensure_worker(self) -> None:
        with self._lock:
            if self._worker is None or not self._worker.is_alive():
                self._worker = threading.Thread(
                    target=self._run_loop, name="openjspace-jobs", daemon=True
                )
                self._worker.start()

    def _run_loop(self) -> None:
        while True:
            item = self._queue.get()
            if item is None:
                return
            job, func = item
            if job.cancel_event.is_set():
                job.status = "cancelled"
                continue
            job.status = "running"
            try:
                job.result = func(job) or {}
                job.status = "succeeded"
            except JobCancelled:
                job.status = "cancelled"
                job.log("job cancelled")
            except Exception as exc:  # error is preserved on the job for the client
                job.status = "failed"
                job.error = f"{type(exc).__name__}: {exc}"
                job.log(traceback.format_exc(limit=10))
                logger.exception("job %s (%s) failed", job.job_id, job.kind)

    def submit(self, kind: str, func: JobFunc, *, total: int = 0) -> Job:
        job = Job(job_id=uuid.uuid4().hex[:12], kind=kind, total=total)
        self._jobs[job.job_id] = job
        self._queue.put((job, func))
        self._ensure_worker()
        return job

    def get(self, job_id: str) -> Job | None:
        return self._jobs.get(job_id)

    def cancel(self, job_id: str) -> Job | None:
        job = self._jobs.get(job_id)
        if job is None:
            return None
        job.cancel_event.set()
        if job.status == "queued":
            job.status = "cancelled"
        return job

    def list(self) -> list[Job]:
        return list(self._jobs.values())
