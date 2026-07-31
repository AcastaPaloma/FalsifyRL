from __future__ import annotations

import json
from pathlib import Path

import pytest

from falsifyrl import SCENARIO_DEFINITIONS, FailureType, generate_paired_cases
from falsifyrl.evaluation import evaluate_completions, load_prediction_jsonl
from falsifyrl.schema import Diagnosis, Verdict


def _cases():
    return generate_paired_cases(
        seeds=(4, 5),
        scenarios=(SCENARIO_DEFINITIONS[-1],),
    )


def _aligned_completion() -> str:
    return Diagnosis(
        verdict=Verdict.ALIGNED,
        failure_type=FailureType.NONE,
        responsible_agents=(),
        evidence_steps=(),
        counterexample_config={},
        reward_patch=None,
        expected_effect="No patch needed.",
        confidence=0.5,
    ).to_json()


def test_oracle_ceiling_scores_one_on_all_metrics() -> None:
    cases = _cases()
    completions = {case.example_id: case.diagnosis.to_json() for case in cases}

    metrics = evaluate_completions(cases, completions)

    assert all(
        value == 1.0
        for key, value in metrics.to_dict().items()
        if key != "example_count"
    )


def test_balanced_pairs_expose_always_aligned_baseline() -> None:
    cases = _cases()
    completions = {case.example_id: _aligned_completion() for case in cases}

    metrics = evaluate_completions(cases, completions)

    assert metrics.json_validity == 1.0
    assert metrics.verdict_accuracy == 0.5
    assert metrics.verdict_macro_f1 == pytest.approx(1 / 3)
    assert metrics.executable_patch_success == 0.5


def test_invalid_or_missing_predictions_score_as_failures() -> None:
    cases = _cases()
    completions = {
        cases[0].example_id: "{not-json",
        cases[1].example_id: cases[1].diagnosis.to_json(),
    }

    metrics = evaluate_completions(cases, completions)

    assert metrics.json_validity == pytest.approx(1 / len(cases))
    assert metrics.composite_score < 0.1


def test_prediction_loader_requires_unique_exact_schema(tmp_path: Path) -> None:
    path = tmp_path / "predictions.jsonl"
    row = {"example_id": "frl-1", "completion": "{}"}
    path.write_text(json.dumps(row) + "\n" + json.dumps(row) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="duplicate prediction"):
        load_prediction_jsonl(path)

