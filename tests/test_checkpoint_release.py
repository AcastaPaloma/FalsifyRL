from __future__ import annotations

import io
import json
import tarfile
from pathlib import Path

import pytest

from falsifyrl.autoscientist import AutoScientistPlan, WorkflowState
from falsifyrl.release import extract_adapter_checkpoint, prepare_model_bundle
from scripts.continue_model_evaluation import await_successful_training


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


def test_checkpoint_preparation_extracts_and_audits_adapter(tmp_path: Path) -> None:
    archive = tmp_path / "checkpoint.tgz"
    bundle = tmp_path / "bundle"
    report = tmp_path / "evaluation.json"
    template = tmp_path / "README.template.md"
    license_file = tmp_path / "LICENSE"
    _checkpoint(archive)
    report.write_text('{"composite_score":0.8}', encoding="utf-8")
    template.write_text(
        "BASE_MODEL_ID DATASET_REPO_ID AUTOSCIENTIST_RUN_ID "
        "BEST_WIN_RATE EVALUATION_REPORT_URL",
        encoding="utf-8",
    )
    license_file.write_text("MIT", encoding="utf-8")

    manifest = prepare_model_bundle(
        archive,
        bundle,
        base_model_id="base/model",
        dataset_repo_id="owner/falsifyrl-seed",
        autoscientist_run_id="run-123",
        best_win_rate=0.81,
        evaluation_report=report,
        model_card_template=template,
        license_path=license_file,
    )

    assert (bundle / "adapter_model.safetensors").read_bytes() == b"safe-tensor-bytes"
    assert manifest["best_win_rate"] == 0.81
    assert "adapter_model.safetensors" in manifest["files"]


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
