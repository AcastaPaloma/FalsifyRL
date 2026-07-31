from __future__ import annotations

import copy

import pytest

from falsifyrl.reporting import REQUIRED_METRICS, build_comparison_report


def _evaluation(composite: float, json_validity: float) -> dict:
    metrics = {name: 0.5 for name in REQUIRED_METRICS}
    metrics.update(
        {
            "example_count": 640,
            "composite_score": composite,
            "json_validity": json_validity,
        }
    )
    return {"split": "test", "source": "predictions.jsonl", "metrics": metrics}


def test_comparison_report_proves_same_split_and_positive_improvement() -> None:
    report = build_comparison_report(
        _evaluation(0.4, 0.8),
        _evaluation(0.7, 0.98),
        base_model_id="Qwen/Qwen3.5-0.8B",
        dataset_manifest_sha256="a" * 64,
        adapter_sha256="b" * 64,
        autoscientist_run_id="run-123",
    )

    assert report.value["submission_thresholds"]["positive_composite_improvement"]
    assert report.value["metrics"]["improvement"]["composite_score"] == pytest.approx(0.3)
    assert "Submission thresholds: **PASS**" in report.to_markdown()


def test_comparison_report_rejects_weak_or_mismatched_evidence() -> None:
    with pytest.raises(ValueError, match="does not improve"):
        build_comparison_report(
            _evaluation(0.7, 0.99),
            _evaluation(0.6, 0.99),
            base_model_id="base",
            dataset_manifest_sha256="a",
            adapter_sha256="b",
            autoscientist_run_id="run",
        )

    adapted = copy.deepcopy(_evaluation(0.8, 0.94))
    with pytest.raises(ValueError, match="below 95%"):
        build_comparison_report(
            _evaluation(0.4, 0.8),
            adapted,
            base_model_id="base",
            dataset_manifest_sha256="a",
            adapter_sha256="b",
            autoscientist_run_id="run",
        )

    adapted["metrics"]["json_validity"] = 0.99
    adapted["metrics"]["example_count"] = 639
    with pytest.raises(ValueError, match="exactly 640"):
        build_comparison_report(
            _evaluation(0.4, 0.8),
            adapted,
            base_model_id="base",
            dataset_manifest_sha256="a",
            adapter_sha256="b",
            autoscientist_run_id="run",
        )
