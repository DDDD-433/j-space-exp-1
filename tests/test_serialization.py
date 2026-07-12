"""Tests for the versioned lens artifact format and compatibility checks."""

import json

import pytest
import torch

from openjspace.core.fitting import fit
from openjspace.core.lens import JacobianLens
from openjspace.core.serialization import (
    LENS_FILENAME,
    METADATA_FILENAME,
    LensCompatibilityError,
    LensMetadata,
    check_compatibility,
    load_lens,
    new_metadata,
    save_lens,
)
from openjspace.models.tiny import TinyAdapter


def _metadata(**overrides) -> LensMetadata:
    defaults = dict(
        model_id="openjspace/tiny-test-model",
        model_architecture="TinyModel",
        tokenizer_id="openjspace/tiny-byte-tokenizer",
        hidden_size=8,
        vocab_size=32,
        n_layers=4,
        source_layers=[0, 1],
        target_layer=3,
    )
    defaults.update(overrides)
    return new_metadata(**defaults)


def test_save_load_roundtrip(tmp_path):
    jacobians = {0: torch.randn(8, 8), 1: torch.randn(8, 8)}
    save_lens(tmp_path / "lens", jacobians, _metadata())
    loaded, metadata = load_lens(tmp_path / "lens")
    assert metadata.format_version == 1
    assert metadata.method == "jacobian_lens"
    assert metadata.created_at and metadata.library_versions
    for layer in (0, 1):
        assert loaded[layer].dtype == torch.float32
        torch.testing.assert_close(loaded[layer], jacobians[layer], rtol=0, atol=2e-3)  # fp16


def test_save_rejects_nonfinite(tmp_path):
    bad = torch.randn(8, 8)
    bad[0, 0] = float("inf")
    with pytest.raises(ValueError, match="NaN/Inf"):
        save_lens(tmp_path / "lens", {0: bad, 1: torch.randn(8, 8)}, _metadata())


def test_load_missing_directory(tmp_path):
    with pytest.raises(FileNotFoundError, match="not a lens artifact"):
        load_lens(tmp_path / "nope")


def test_load_rejects_corrupted_metadata(tmp_path):
    save_lens(tmp_path / "lens", {0: torch.randn(8, 8), 1: torch.randn(8, 8)}, _metadata())
    meta_path = tmp_path / "lens" / METADATA_FILENAME
    meta = json.loads(meta_path.read_text())
    meta["source_layers"] = [0, 5]  # disagrees with stored tensors
    meta_path.write_text(json.dumps(meta))
    with pytest.raises(ValueError, match="disagree"):
        load_lens(tmp_path / "lens")


def test_load_rejects_wrong_shape(tmp_path):
    save_lens(tmp_path / "lens", {0: torch.randn(8, 8), 1: torch.randn(8, 8)}, _metadata())
    meta_path = tmp_path / "lens" / METADATA_FILENAME
    meta = json.loads(meta_path.read_text())
    meta["hidden_size"] = 16
    meta_path.write_text(json.dumps(meta))
    with pytest.raises(ValueError, match="shape"):
        load_lens(tmp_path / "lens")


def test_load_rejects_newer_format(tmp_path):
    save_lens(tmp_path / "lens", {0: torch.randn(8, 8), 1: torch.randn(8, 8)}, _metadata())
    meta_path = tmp_path / "lens" / METADATA_FILENAME
    meta = json.loads(meta_path.read_text())
    meta["format_version"] = 99
    meta_path.write_text(json.dumps(meta))
    with pytest.raises(ValueError, match="format_version"):
        load_lens(tmp_path / "lens")


def test_compatibility_check_pass_and_fail():
    metadata = _metadata()
    ok = check_compatibility(
        metadata,
        model_id="openjspace/tiny-test-model",
        architecture="TinyModel",
        hidden_size=8,
        vocab_size=32,
        n_layers=4,
        tokenizer_id="openjspace/tiny-byte-tokenizer",
        residual_location="block_output",
    )
    assert ok == []
    with pytest.raises(LensCompatibilityError, match="hidden_size"):
        check_compatibility(
            metadata,
            model_id="openjspace/tiny-test-model",
            architecture="TinyModel",
            hidden_size=16,
            vocab_size=32,
            n_layers=4,
            tokenizer_id="openjspace/tiny-byte-tokenizer",
            residual_location="block_output",
        )


def test_compatibility_residual_location_mismatch():
    metadata = _metadata(residual_location="block_output")
    with pytest.raises(LensCompatibilityError, match="residual_location"):
        check_compatibility(
            metadata,
            model_id="openjspace/tiny-test-model",
            architecture="TinyModel",
            hidden_size=8,
            vocab_size=32,
            n_layers=4,
            tokenizer_id="openjspace/tiny-byte-tokenizer",
            residual_location="block_input",
        )


def test_compatibility_force_returns_warnings():
    metadata = _metadata()
    warnings = check_compatibility(
        metadata,
        model_id="some/other-model",
        architecture="TinyModel",
        hidden_size=8,
        vocab_size=32,
        n_layers=4,
        tokenizer_id="openjspace/tiny-byte-tokenizer",
        residual_location="block_output",
        force=True,
    )
    assert warnings and "model_id" in warnings[0]


def test_fitted_lens_roundtrip_through_artifact(tmp_path):
    adapter = TinyAdapter()
    lens = fit(adapter, ["abcdefghij " * 5], source_layers=[0, 2], dim_batch=4, max_seq_len=64)
    lens.save(tmp_path / "artifact")
    assert (tmp_path / "artifact" / LENS_FILENAME).is_file()
    reloaded = JacobianLens.load(tmp_path / "artifact")
    assert reloaded.validate_against(adapter) == []
    assert reloaded.source_layers == [0, 2]
    for layer in (0, 2):
        torch.testing.assert_close(
            reloaded.jacobians[layer], lens.jacobians[layer], rtol=0, atol=2e-3
        )


def test_json_export_roundtrip(tmp_path):
    from openjspace.core.applying import inspect_prompt
    from openjspace.report.schema import RunResult

    adapter = TinyAdapter()
    lens = fit(adapter, ["abcdefghij " * 5], source_layers=[0], dim_batch=4, max_seq_len=64)
    result = inspect_prompt(adapter, lens, "the quick brown fox", top_k=3)
    path = tmp_path / "run.json"
    path.write_text(result.model_dump_json())
    reloaded = RunResult.model_validate_json(path.read_text())
    assert reloaded == result
