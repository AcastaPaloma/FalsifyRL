from __future__ import annotations

from collections import defaultdict

from falsifyrl import (
    SCENARIO_DEFINITIONS,
    Diagnosis,
    FailureType,
    generate_case,
    generate_cases,
    verify_case,
)


def test_scenario_families_are_exclusive_to_one_split() -> None:
    splits_by_family: dict[str, set[str]] = defaultdict(set)
    for scenario in SCENARIO_DEFINITIONS:
        splits_by_family[scenario.family].add(scenario.split.value)

    assert all(len(splits) == 1 for splits in splits_by_family.values())
    assert {scenario.split.value for scenario in SCENARIO_DEFINITIONS} == {
        "train",
        "validation",
        "test",
    }


def test_generation_is_deterministic() -> None:
    scenario = SCENARIO_DEFINITIONS[0]

    first = generate_case(scenario, FailureType.FREE_RIDING, seed=42)
    second = generate_case(scenario, FailureType.FREE_RIDING, seed=42)

    assert first == second
    assert first.training_record() == second.training_record()


def test_every_failure_family_passes_executable_verification() -> None:
    cases = generate_cases(seeds=range(12))

    assert len(cases) == len(SCENARIO_DEFINITIONS) * len(FailureType) * 12
    for case in cases:
        result = verify_case(case)
        assert result.valid, (case.example_id, result.errors)


def test_prompts_do_not_leak_generation_or_oracle_metadata() -> None:
    forbidden_markers = {
        "gold_label",
        "injected_defect",
        "expected_verdict",
        "true_success",
        "validator_result",
        "generator_seed",
    }

    for case in generate_cases(seeds=range(3)):
        prompt = case.render_prompt().lower()
        assert not any(marker in prompt for marker in forbidden_markers)
        assert f"seed: {case.seed}" not in prompt
        assert case.example_id not in prompt


def test_all_completions_parse_against_strict_schema() -> None:
    for case in generate_cases(seeds=range(3)):
        parsed = Diagnosis.from_json(case.training_record()["completion"])
        assert parsed == case.diagnosis
