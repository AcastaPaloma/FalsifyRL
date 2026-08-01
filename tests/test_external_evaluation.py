from __future__ import annotations

import json
from pathlib import Path

import pytest

from falsifyrl.autoscientist import AutoScientistPlan, WorkflowState
from falsifyrl.dataset import DatasetBuildConfig, build_cases
from scripts.finalize_external_evaluation import (
    evaluate_exact_predictions,
    finalize,
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
