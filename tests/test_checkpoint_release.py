from __future__ import annotations

import io
import json
import os
import tarfile
from pathlib import Path

import pytest
import zstandard

from falsifyrl.autoscientist import AutoScientistPlan, WorkflowState
from falsifyrl.release import (
    canonicalize_adapter_base_model,
    extract_adapter_checkpoint,
    prepare_model_bundle,
)
from scripts.continue_model_evaluation import (
    _subprocess_environment,
    await_successful_training,
)


def _checkpoint(path: Path) -> None:
    adapter_config = json.dumps({"base_model_name_or_path": "base/model"}).encode()
    weights = b"safe-tensor-bytes"
    with tarfile.open(path, "w:gz") as archive:
        for name, content in (
            ("checkpoint/adapter_config.json", adapter_config),
            ("checkpoint/adapter_model.safetensors", weights),
            ("checkpoint/tokenizer.json", b"{}"),
        ):
            info = tarfile.TarInfo(name)
            info.size = len(content)
            archive.addfile(info, io.BytesIO(content))


def _zstandard_checkpoint(path: Path) -> None:
    uncompressed = io.BytesIO()
    adapter_config = json.dumps({"base_model_name_or_path": "base/model"}).encode()
    with tarfile.open(fileobj=uncompressed, mode="w") as archive:
        for name, content in (
            ("checkpoint/adapter_config.json", adapter_config),
            ("checkpoint/adapter_model.safetensors", b"safe-tensor-bytes"),
        ):
            info = tarfile.TarInfo(name)
            info.size = len(content)
            archive.addfile(info, io.BytesIO(content))
    path.write_bytes(zstandard.ZstdCompressor().compress(uncompressed.getvalue()))


def test_checkpoint_preparation_extracts_and_audits_adapter(tmp_path: Path) -> None:
    archive = tmp_path / "checkpoint.tgz"
    bundle = tmp_path / "bundle"
    report = tmp_path / "evaluation.json"
    template = tmp_path / "README.template.md"
    license_file = tmp_path / "LICENSE"
    base_predictions = tmp_path / "base.jsonl"
    adapted_predictions = tmp_path / "adapted.jsonl"
    evaluation_metadata = tmp_path / "evidence"
    _checkpoint(archive)
    report.write_text('{"composite_score":0.8}', encoding="utf-8")
    template.write_text(
        "BASE_MODEL_ID DATASET_REPO_ID AUTOSCIENTIST_RUN_ID "
        "BEST_WIN_RATE EVALUATION_REPORT_URL",
        encoding="utf-8",
    )
    license_file.write_text("MIT", encoding="utf-8")
    base_predictions.write_text('{"example_id":"a","completion":"{}"}\n')
    adapted_predictions.write_text('{"example_id":"a","completion":"{}"}\n')
    evaluation_metadata.mkdir()
    (evaluation_metadata / "evaluation-manifest.json").write_text("{}\n")
    (evaluation_metadata / "colab-evaluation.json").write_text("{}\n")

    manifest = prepare_model_bundle(
        archive,
        bundle,
        base_model_id="base/model",
        dataset_repo_id="owner/falsifyrl-seed",
        autoscientist_run_id="run-123",
        best_win_rate=0.81,
        evaluation_report=report,
        base_predictions=base_predictions,
        adapted_predictions=adapted_predictions,
        evaluation_metadata_dir=evaluation_metadata,
        model_card_template=template,
        license_path=license_file,
    )

    assert (bundle / "adapter_model.safetensors").read_bytes() == b"safe-tensor-bytes"
    assert manifest["best_win_rate"] == 0.81
    assert "adapter_model.safetensors" in manifest["files"]
    assert manifest["files"]["falsifyrl-base-test-predictions.jsonl"][
        "sha256"
    ]
    assert manifest["files"]["falsifyrl-adapted-test-predictions.jsonl"][
        "sha256"
    ]
    assert "evaluation-manifest.json" in manifest["files"]
    assert "colab-evaluation.json" in manifest["files"]


def test_checkpoint_can_be_extracted_for_prepublication_evaluation(
    tmp_path: Path,
) -> None:
    archive = tmp_path / "checkpoint.tgz"
    _checkpoint(archive)

    adapter_root = extract_adapter_checkpoint(archive, tmp_path / "extracted")

    assert adapter_root.name == "checkpoint"
    assert (adapter_root / "adapter_model.safetensors").read_bytes() == (
        b"safe-tensor-bytes"
    )


def test_adapter_base_model_alias_is_canonicalized_by_matching_slug(
    tmp_path: Path,
) -> None:
    adapter_root = tmp_path / "adapter"
    adapter_root.mkdir()
    config_path = adapter_root / "adapter_config.json"
    config_path.write_text(
        json.dumps(
            {"base_model_name_or_path": "togethercomputer/Qwen3.5-0.8B"}
        ),
        encoding="utf-8",
    )

    original = canonicalize_adapter_base_model(
        adapter_root,
        "Qwen/Qwen3.5-0.8B",
    )

    assert original == "togethercomputer/Qwen3.5-0.8B"
    assert json.loads(config_path.read_text(encoding="utf-8"))[
        "base_model_name_or_path"
    ] == "Qwen/Qwen3.5-0.8B"


def test_decorated_internal_llama_alias_is_canonicalized(
    tmp_path: Path,
) -> None:
    adapter_root = tmp_path / "adapter"
    adapter_root.mkdir()
    config_path = adapter_root / "adapter_config.json"
    config_path.write_text(
        json.dumps(
            {
                "base_model_name_or_path": (
                    "togethercomputer/"
                    "Meta-Llama-3.2-3B-Instruct-Reference__TOG__FT"
                )
            }
        ),
        encoding="utf-8",
    )

    original = canonicalize_adapter_base_model(
        adapter_root,
        "meta-llama/Llama-3.2-3B-Instruct",
    )

    assert original.endswith("Meta-Llama-3.2-3B-Instruct-Reference__TOG__FT")
    assert json.loads(config_path.read_text(encoding="utf-8"))[
        "base_model_name_or_path"
    ] == "meta-llama/Llama-3.2-3B-Instruct"


def test_adapter_base_model_canonicalization_rejects_different_model(
    tmp_path: Path,
) -> None:
    adapter_root = tmp_path / "adapter"
    adapter_root.mkdir()
    (adapter_root / "adapter_config.json").write_text(
        json.dumps({"base_model_name_or_path": "internal/different-model"}),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="does not match"):
        canonicalize_adapter_base_model(
            adapter_root,
            "Qwen/Qwen3.5-0.8B",
        )


def test_zstandard_checkpoint_can_be_extracted_for_prepublication_evaluation(
    tmp_path: Path,
) -> None:
    archive = tmp_path / "checkpoint.tgz"
    _zstandard_checkpoint(archive)

    adapter_root = extract_adapter_checkpoint(archive, tmp_path / "extracted")

    assert adapter_root.name == "checkpoint"
    assert (adapter_root / "adapter_model.safetensors").read_bytes() == (
        b"safe-tensor-bytes"
    )


def test_model_evaluation_waits_for_downloadable_successful_run(
    tmp_path: Path,
) -> None:
    state_path = tmp_path / "workflow.json"
    WorkflowState(
        plan=AutoScientistPlan(
            source="file",
            local_file="train.jsonl",
            expected_training_rows=1,
        ),
        autoscientist_run_id="experiment-123",
        autoscientist_status="succeeded",
        best_win_rate=0.8,
        download_available=True,
    ).save(state_path)

    state = await_successful_training(
        state_path,
        poll_seconds=0,
        timeout_seconds=1,
    )

    assert state.autoscientist_run_id == "experiment-123"
    assert state.best_win_rate == 0.8


def test_model_evaluation_exposes_repository_source_to_isolated_python(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("PYTHONPATH", "existing-path")

    environment = _subprocess_environment(tmp_path)

    assert environment["PYTHONPATH"].split(os.pathsep) == [
        str((tmp_path / "src").resolve()),
        "existing-path",
    ]


def test_checkpoint_preparation_rejects_path_traversal(tmp_path: Path) -> None:
    archive = tmp_path / "unsafe.tgz"
    with tarfile.open(archive, "w:gz") as bundle:
        payload = b"escape"
        info = tarfile.TarInfo("../outside.txt")
        info.size = len(payload)
        bundle.addfile(info, io.BytesIO(payload))

    with pytest.raises(ValueError, match="unsafe checkpoint archive path"):
        prepare_model_bundle(
            archive,
            tmp_path / "bundle",
            base_model_id="base/model",
            dataset_repo_id="owner/data",
            autoscientist_run_id="run-1",
            best_win_rate=0.8,
            evaluation_report=tmp_path / "missing.json",
        )
