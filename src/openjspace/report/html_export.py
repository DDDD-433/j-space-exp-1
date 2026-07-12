"""Self-contained HTML report export.

Renders a :class:`~openjspace.report.schema.RunResult` into a single HTML file
with the run JSON embedded inline and no network dependencies (no CDN
fetches), suitable for archiving and sharing.
"""

from __future__ import annotations

import html
from importlib.resources import files
from pathlib import Path

from openjspace.report.schema import RunResult


def _template() -> str:
    return (files("openjspace.report") / "templates" / "report.html").read_text(encoding="utf-8")


def render_html(result: RunResult, *, title: str | None = None) -> str:
    """Render the run to a self-contained HTML string."""
    if title is None:
        prompt = result.metadata.prompt
        short = prompt[:60] + ("…" if len(prompt) > 60 else "")
        title = f"OpenJSpace report — {short}"
    payload = result.model_dump_json()
    # ``</`` -> ``<\/`` so token strings can't close the <script> tag.
    payload = payload.replace("</", "<\\/")
    return _template().replace("__TITLE__", html.escape(title)).replace("__RUN_JSON__", payload)


def export_html(result: RunResult, path: str | Path, *, title: str | None = None) -> Path:
    """Write the self-contained HTML report to ``path``."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_html(result, title=title), encoding="utf-8")
    return path
