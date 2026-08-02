from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import pytest

from falsifyrl.autoscientist import AutoScientistPlan, WorkflowState
from falsifyrl.dataset import DatasetBuildConfig, build_cases
from scripts import finalize_external_evaluation
from scripts.finalize_external_evaluation import (
    evaluate_exact_predictions,
    finalize,
    verify_staged_evidence,
)


def _write_predictions(path: Path, completions: dict[str, str]) -> None:
    path.write_text(
        "".join(
            json.dumps({"example_id": key, "completion": value}) + "\n"
            for key, value in completions.items()
        ),
        encoding="utf-8",
    )


def _test_cases() -> list:
    return [
        case
        for case in build_cases(DatasetBuildConfig())
        if case.scenario.split.value == "test"
    ]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_external_predictions_must_cover_exact_test_split(tmp_path: Path) -> None:
    cases = _test_cases()
    path = tmp_path / "predictions.jsonl"
    _write_predictions(
        path,
        {case.example_id: case.diagnosis.to_json() for case in cases[:-1]},
    )

    with pytest.raises(ValueError, match="missing=1"):
        evaluate_exact_predictions(path)


def test_finalize_builds_fail_closed_colab_comparison(tmp_path: Path) -> None:
    cases = _test_cases()
    aligned = next(case.diagnosis for case in cases if case.case_role == "control")
    base_predictions = tmp_path / "base.jsonl"
    adapted_predictions = tmp_path / "adapted.jsonl"
    _write_predictions(
        base_predictions,
        {case.example_id: aligned.to_json() for case in cases},
    )
    _write_predictions(
        adapted_predictions,
        {case.example_id: case.diagnosis.to_json() for case in cases},
    )

    state = WorkflowState(
        plan=AutoScientistPlan(
            source="file",
            local_file="train.jsonl",
            model="Qwen/Qwen3.5-9B",
        ),
        autoscientist_run_id="run-qwen",
        autoscientist_status="succeeded",
        best_win_rate=0.9,
        resolved_model="Qwen/Qwen3.5-9B",
        download_available=True,
    )
    state_path = tmp_path / "workflow.json"
    state.save(state_path)
    adapter = tmp_path / "adapter_model.safetensors"
    adapter.write_bytes(b"adapter")
    dataset_manifest = tmp_path / "dataset-manifest.json"
    dataset_manifest.write_text("{}\n", encoding="utf-8")
    submission = tmp_path / "submission.json"
    submission.write_text(
        json.dumps(
            {
                "identifiers": {},
                "metrics": {},
            }
        ),
        encoding="utf-8",
    )

    comparison = finalize(
        state_path=state_path,
        base_predictions=base_predictions,
        adapted_predictions=adapted_predictions,
        adapter_weights=adapter,
        dataset_manifest=dataset_manifest,
        base_report_path=tmp_path / "base-report.json",
        adapted_report_path=tmp_path / "adapted-report.json",
        comparison_json_path=tmp_path / "comparison.json",
        comparison_markdown_path=tmp_path / "comparison.md",
        submission_manifest_path=submission,
    )

    assert comparison["metrics"]["adapted"]["composite_score"] == 1.0
    assert comparison["metrics"]["improvement"]["composite_score"] > 0
    assert comparison["evidence"]["autoscientist_run_id"] == "run-qwen"
    updated = json.loads(submission.read_text(encoding="utf-8"))
    assert updated["identifiers"]["base_model_id"] == "Qwen/Qwen3.5-9B"
    assert updated["metrics"]["trained_json_validity"] == 1.0


def test_final_comparison_preserves_immutable_evidence_revision(tmp_path: Path) -> None:
    cases = _test_cases()
    aligned = next(case.diagnosis for case in cases if case.case_role == "control")
    base_predictions = tmp_path / "base.jsonl"
    adapted_predictions = tmp_path / "adapted.jsonl"
    _write_predictions(
        base_predictions,
        {case.example_id: aligned.to_json() for case in cases},
    )
    _write_predictions(
        adapted_predictions,
        {case.example_id: case.diagnosis.to_json() for case in cases},
    )
    state = WorkflowState(
        plan=AutoScientistPlan(source="file", local_file="train.jsonl"),
        autoscientist_run_id="run-immutable",
        autoscientist_status="succeeded",
        best_win_rate=0.9,
        resolved_model="base/model",
        download_available=True,
    )
    state_path = tmp_path / "workflow.json"
    state.save(state_path)
    adapter = tmp_path / "adapter.safetensors"
    adapter.write_bytes(b"adapter")
    dataset_manifest = tmp_path / "dataset.json"
    dataset_manifest.write_text("{}\n")

    comparison = finalize(
        state_path=state_path,
        base_predictions=base_predictions,
        adapted_predictions=adapted_predictions,
        adapter_weights=adapter,
        dataset_manifest=dataset_manifest,
        base_report_path=tmp_path / "base-report.json",
        adapted_report_path=tmp_path / "adapted-report.json",
        comparison_json_path=tmp_path / "comparison.json",
        comparison_markdown_path=tmp_path / "comparison.md",
        submission_manifest_path=None,
        evidence_provenance={
            "staging_repo_id": "owner/private-evidence",
            "evidence_revision": "a" * 40,
            "checkpoint_revision": "b" * 40,
        },
    )

    assert comparison["evidence"]["evidence_revision"] == "a" * 40
    assert comparison["evidence"]["checkpoint_revision"] == "b" * 40


def test_staged_evidence_is_bound_to_exact_run_and_files(tmp_path: Path) -> None:
    cases = _test_cases()
    aligned = next(case.diagnosis for case in cases if case.case_role == "control")
    evidence = tmp_path / "evidence"
    evidence.mkdir()
    base = evidence / "falsifyrl-base-test-predictions.jsonl"
    adapted = evidence / "falsifyrl-adapted-test-predictions.jsonl"
    _write_predictions(
        base,
        {case.example_id: aligned.to_json() for case in cases},
    )
    _write_predictions(
        adapted,
        {case.example_id: case.diagnosis.to_json() for case in cases},
    )

    state = WorkflowState(
        plan=AutoScientistPlan(
            source="file",
            local_file="train.jsonl",
            model="Qwen/Qwen3.5-9B",
        ),
        autoscientist_run_id="run-qwen",
        autoscientist_status="succeeded",
        best_win_rate=0.9,
        resolved_model="Qwen/Qwen3.5-9B",
        download_available=True,
    )
    state_path = tmp_path / "workflow.json"
    state.save(state_path)
    adapter = tmp_path / "adapter_model.safetensors"
    adapter.write_bytes(b"adapter")
    dataset_manifest = tmp_path / "dataset-manifest.json"
    dataset_manifest.write_text(
        json.dumps(
            {"files": {"test.jsonl": {"sha256": "heldout-test-sha"}}}
        ),
        encoding="utf-8",
    )
    report = {
        "run_id": "run-qwen",
        "base_model_id": "Qwen/Qwen3.5-9B",
        "adapter_sha256": _sha256(adapter),
        "example_count": 640,
        "base_predictions_sha256": _sha256(base),
        "adapted_predictions_sha256": _sha256(adapted),
    }
    report_path = evidence / "colab-evaluation.json"
    report_path.write_text(json.dumps(report), encoding="utf-8")
    files = {
        path.name: {"sha256": _sha256(path), "bytes": path.stat().st_size}
        for path in (report_path, base, adapted)
    }
    manifest = {
        "schema_version": 1,
        "autoscientist_run_id": "run-qwen",
        "base_model_id": "Qwen/Qwen3.5-9B",
        "checkpoint_revision": "a" * 40,
        "adapter_sha256": _sha256(adapter),
        "test_jsonl_sha256": "heldout-test-sha",
        "example_count": 640,
        "do_sample": False,
        "files": files,
    }
    (evidence / "evaluation-manifest.json").write_text(
        json.dumps(manifest),
        encoding="utf-8",
    )

    verified = verify_staged_evidence(
        evidence_dir=evidence,
        state_path=state_path,
        adapter_weights=adapter,
        dataset_manifest=dataset_manifest,
        checkpoint_revision="a" * 40,
    )
    assert verified == (base, adapted)

    with pytest.raises(ValueError, match="checkpoint revision"):
        verify_staged_evidence(
            evidence_dir=evidence,
            state_path=state_path,
            adapter_weights=adapter,
            dataset_manifest=dataset_manifest,
            checkpoint_revision="b" * 40,
        )

    adapted.write_text("{}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="hash mismatch"):
        verify_staged_evidence(
            evidence_dir=evidence,
            state_path=state_path,
            adapter_weights=adapter,
            dataset_manifest=dataset_manifest,
            checkpoint_revision="a" * 40,
        )


def test_main_loads_dotenv_before_reading_staging_token(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = WorkflowState(
        plan=AutoScientistPlan(source="file", local_file="train.jsonl"),
        autoscientist_run_id="run-qwen",
    )
    state_path = tmp_path / "workflow.json"
    state.save(state_path)
    args = finalize_external_evaluation.argparse.Namespace(
        state=state_path,
        base_predictions=None,
        adapted_predictions=None,
        staging_repo_id="owner/private-staging",
        evidence_revision="b" * 40,
        checkpoint_revision="a" * 40,
        adapter_weights=tmp_path / "adapter_model.safetensors",
        dataset_manifest=tmp_path / "dataset-manifest.json",
        output_dir=tmp_path / "output",
        submission_manifest=tmp_path / "submission.json",
    )
    captured: dict[str, str] = {}

    def fake_load_dotenv() -> bool:
        os.environ["HF_TOKEN"] = "loaded-from-dotenv"
        return True

    def fake_download(**kwargs: object) -> Path:
        captured["token"] = str(kwargs["token"])
        captured["revision"] = str(kwargs["revision"])
        raise RuntimeError("stop after token capture")

    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.setattr(finalize_external_evaluation, "load_dotenv", fake_load_dotenv)
    monkeypatch.setattr(finalize_external_evaluation, "parse_args", lambda: args)
    monkeypatch.setattr(
        finalize_external_evaluation,
        "download_staged_evidence",
        fake_download,
    )

    with pytest.raises(RuntimeError, match="stop after token capture"):
        finalize_external_evaluation.main()

    assert captured["token"] == "loaded-from-dotenv"
    assert captured["revision"] == "b" * 40


def test_main_keeps_evidence_and_checkpoint_revisions_distinct(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = WorkflowState(
        plan=AutoScientistPlan(source="file", local_file="train.jsonl"),
        autoscientist_run_id="run-qwen",
    )
    state_path = tmp_path / "workflow.json"
    state.save(state_path)
    evidence_revision = "b" * 40
    checkpoint_revision = "a" * 40
    args = finalize_external_evaluation.argparse.Namespace(
        state=state_path,
        base_predictions=None,
        adapted_predictions=None,
        staging_repo_id="owner/private-staging",
        evidence_revision=evidence_revision,
        checkpoint_revision=checkpoint_revision,
        adapter_weights=tmp_path / "adapter_model.safetensors",
        dataset_manifest=tmp_path / "dataset-manifest.json",
        output_dir=tmp_path / "output",
        submission_manifest=tmp_path / "submission.json",
    )
    staged_dir = tmp_path / "staged"
    captured: dict[str, str] = {}

    def fake_download(**kwargs: object) -> Path:
        captured["evidence_revision"] = str(kwargs["revision"])
        return staged_dir

    def fake_verify(**kwargs: object) -> tuple[Path, Path]:
        captured["checkpoint_revision"] = str(kwargs["checkpoint_revision"])
        return tmp_path / "base.jsonl", tmp_path / "adapted.jsonl"

    def fake_finalize(**_: object) -> dict:
        return {"ok": True}

    monkeypatch.setattr(finalize_external_evaluation, "load_dotenv", lambda: True)
    monkeypatch.setattr(finalize_external_evaluation, "parse_args", lambda: args)
    monkeypatch.setattr(
        finalize_external_evaluation,
        "download_staged_evidence",
        fake_download,
    )
    monkeypatch.setattr(
        finalize_external_evaluation,
        "verify_staged_evidence",
        fake_verify,
    )
    monkeypatch.setattr(finalize_external_evaluation, "finalize", fake_finalize)

    finalize_external_evaluation.main()

    assert captured == {
        "evidence_revision": evidence_revision,
        "checkpoint_revision": checkpoint_revision,
    }
