"""API tests over the tiny model (no downloads)."""

import io
import json
import time

import pytest
from fastapi.testclient import TestClient

from openjspace.core.fitting import fit
from openjspace.models.tiny import TINY_MODEL_ID, TinyAdapter


@pytest.fixture()
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENJSPACE_HOME", str(tmp_path))
    from openjspace.server.app import create_app

    return TestClient(create_app())


@pytest.fixture()
def tiny_lens_artifact(tmp_path, monkeypatch, client):
    """A fitted tiny lens saved into the server's artifacts directory."""
    adapter = TinyAdapter()
    lens = fit(
        adapter,
        ["abcdefghij " * 5, "klmnopqrst " * 5],
        source_layers=[0, 1, 2],
        dim_batch=4,
        max_seq_len=64,
    )
    lens.save(tmp_path / "artifacts" / "tiny-lens")
    return "tiny-lens"


def _wait_for_job(client, job_id: str, timeout: float = 30.0) -> dict:
    deadline = time.time() + timeout
    while time.time() < deadline:
        info = client.get(f"/api/jobs/{job_id}").json()
        if info["status"] in ("succeeded", "failed", "cancelled"):
            return info
        time.sleep(0.05)
    raise TimeoutError(f"job {job_id} did not finish")


def test_load_model_info(client):
    response = client.post("/api/models/load", json={"model_id": TINY_MODEL_ID})
    assert response.status_code == 200
    info = response.json()
    assert info["n_layers"] == 4
    assert info["hidden_size"] == 8
    assert info["kind"] == "text"
    assert info["residual_location"] == "block_output"


def test_model_families_endpoint(client):
    response = client.get("/api/models/families")
    assert response.status_code == 200
    assert any(f["kind"] == "vlm" for f in response.json())


def test_fit_job_lifecycle(client, tmp_path):
    corpus = tmp_path / "corpus.jsonl"
    corpus.write_text(
        "\n".join(json.dumps({"text": "abcdefghij klmnopqrst uvwxyz " * 8}) for _ in range(3))
    )
    response = client.post(
        "/api/lenses/fit",
        json={
            "model_id": TINY_MODEL_ID,
            "dataset": str(corpus),
            "artifact_name": "fitted-by-api",
            "layers": [0, 1],
            "max_seq_len": 64,
            "num_prompts": 3,
            "dim_batch": 4,
        },
    )
    assert response.status_code == 200
    job = response.json()
    info = _wait_for_job(client, job["job_id"])
    assert info["status"] == "succeeded", info
    assert info["result"]["artifact_name"] == "fitted-by-api"
    artifacts = client.get("/api/artifacts").json()
    assert any(a["name"] == "fitted-by-api" for a in artifacts)


def test_job_cancellation(client, tmp_path):
    corpus = tmp_path / "corpus.jsonl"
    corpus.write_text(
        "\n".join(json.dumps({"text": "abcdefghij klmnopqrst uvwxyz " * 8}) for _ in range(50))
    )
    response = client.post(
        "/api/lenses/fit",
        json={
            "model_id": TINY_MODEL_ID,
            "dataset": str(corpus),
            "artifact_name": "cancelled-fit",
            "layers": [0],
            "max_seq_len": 64,
            "num_prompts": 50,
            "dim_batch": 4,
        },
    )
    job_id = response.json()["job_id"]
    cancel = client.post(f"/api/jobs/{job_id}/cancel")
    assert cancel.status_code == 200
    info = _wait_for_job(client, job_id)
    assert info["status"] == "cancelled"


def test_job_not_found(client):
    assert client.get("/api/jobs/nonexistent0").status_code == 404
    assert client.post("/api/jobs/nonexistent0/cancel").status_code == 404


def test_inspect_endpoint_and_run_reopen(client, tiny_lens_artifact):
    response = client.post(
        "/api/inspect",
        json={
            "model_id": TINY_MODEL_ID,
            "lens_name": tiny_lens_artifact,
            "prompt": "the quick brown fox jumps",
            "top_k": 5,
        },
    )
    assert response.status_code == 200, response.text
    run = response.json()
    assert run["cells"]
    assert run["metadata"]["disclaimer"]
    run_id = run["metadata"]["run_id"]
    # Run artifacts can be reopened.
    reopened = client.get(f"/api/runs/{run_id}")
    assert reopened.status_code == 200
    assert reopened.json() == run
    runs = client.get("/api/runs").json()
    assert any(r["run_id"] == run_id for r in runs)
    report = client.get(f"/api/runs/{run_id}/report.html")
    assert report.status_code == 200
    assert "OpenJSpace report" in report.text


def test_inspect_missing_lens_rejected(client):
    response = client.post(
        "/api/inspect",
        json={
            "model_id": TINY_MODEL_ID,
            "lens_name": "does-not-exist",
            "prompt": "hello world",
        },
    )
    assert response.status_code == 400


def test_decompose_endpoint(client, tiny_lens_artifact):
    response = client.post(
        "/api/decompose",
        json={
            "model_id": TINY_MODEL_ID,
            "lens_name": tiny_lens_artifact,
            "prompt": "the quick brown fox jumps",
            "layer": 2,
            "position": -1,
            "k": 4,
        },
    )
    assert response.status_code == 200, response.text
    record = response.json()
    assert len(record["entries"]) <= 4
    assert all(e["coefficient"] >= 0 for e in record["entries"])
    assert "non-unique" in record["warning"]


def test_malformed_paths_rejected(client):
    for bad in ("../etc", "a/../../b", ".hidden", "sp ace"):
        response = client.post(
            "/api/inspect",
            json={"model_id": TINY_MODEL_ID, "lens_name": bad, "prompt": "hello"},
        )
        assert response.status_code in (400, 422), bad
    assert client.get("/api/runs/..%2F..%2Fetc").status_code in (400, 404)


def test_upload_validation(client):
    # Wrong extension rejected.
    response = client.post(
        "/api/uploads", files={"file": ("evil.sh", io.BytesIO(b"#!/bin/sh"), "text/plain")}
    )
    assert response.status_code == 400
    # Non-image content with image extension rejected.
    response = client.post(
        "/api/uploads", files={"file": ("fake.png", io.BytesIO(b"not an image"), "image/png")}
    )
    assert response.status_code == 400
    # A real image is accepted and served back.
    from PIL import Image

    buffer = io.BytesIO()
    Image.new("RGB", (32, 16), (200, 30, 60)).save(buffer, format="PNG")
    response = client.post(
        "/api/uploads", files={"file": ("ok.png", io.BytesIO(buffer.getvalue()), "image/png")}
    )
    assert response.status_code == 200, response.text
    info = response.json()
    assert (info["width"], info["height"]) == (32, 16)
    served = client.get(f"/api/uploads/{info['name']}")
    assert served.status_code == 200


def test_api_schemas_validate(client):
    # Missing required fields -> 422 from Pydantic validation.
    assert client.post("/api/inspect", json={}).status_code == 422
    assert client.post("/api/lenses/fit", json={"model_id": "x"}).status_code == 422
    assert (
        client.post("/api/lenses/merge", json={"shard_names": [], "output_name": "x"}).status_code
        == 422
    )
