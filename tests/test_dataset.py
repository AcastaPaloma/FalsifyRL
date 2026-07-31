from __future__ import annotations

import csv
import json
from pathlib import Path

from falsifyrl import SCENARIO_DEFINITIONS, FailureType, generate_paired_cases
from falsifyrl.dataset import (
    DatasetBuildConfig,
    build_cases,
    validate_cases,
    write_dataset,
)


def test_paired_cases_share_reward_but_require_trace_reasoning() -> None:
    scenario = SCENARIO_DEFINITIONS[0]
    cases = generate_paired_cases(seeds=(7,), scenarios=(scenario,))

    assert len(cases) == 2 * (len(FailureType) - 1)
    pairs: dict[str, list] = {}
    for case in cases:
        pairs.setdefault(case.pair_id, []).append(case)

    for pair in pairs.values():
        control = next(case for case in pair if case.case_role == "control")
        exploit = next(case for case in pair if case.case_role == "exploit")
        assert control.reward_spec == exploit.reward_spec
        assert control.render_prompt() != exploit.render_prompt()
        assert control.diagnosis.verdict != exploit.diagnosis.verdict


def test_dataset_build_is_balanced_and_family_disjoint() -> None:
    config = DatasetBuildConfig(
        train_seed_count=3,
        validation_seed_count=2,
        test_seed_count=2,
    )
    cases = build_cases(config)
    summary = validate_cases(cases)

    assert summary["case_count"] == 160
    assert summary["pair_count"] == 80
    assert summary["verdict_counts"] == {"aligned": 80, "reward_hack": 80}
    assert summary["families_by_split"] == {
        "test": ["crossing_navigation"],
        "train": ["dual_arm_workspace", "warehouse_handoff"],
        "validation": ["cooperative_transport"],
    }


def test_writer_produces_deterministic_jsonl_and_autoscientist_csv(
    tmp_path: Path,
) -> None:
    config = DatasetBuildConfig(
        train_seed_count=2,
        validation_seed_count=1,
        test_seed_count=1,
    )
    first_dir = tmp_path / "first"
    second_dir = tmp_path / "second"

    first_manifest = write_dataset(first_dir, config=config)
    second_manifest = write_dataset(second_dir, config=config)

    assert first_manifest == second_manifest
    assert first_manifest["validation"]["all_cases_verified"] is True
    assert first_manifest["files"]["train.jsonl"]["sha256"] == second_manifest["files"][
        "train.jsonl"
    ]["sha256"]

    with (first_dir / "train.csv").open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        rows = list(reader)
    assert reader.fieldnames == ["prompt", "completion"]
    assert len(rows) == first_manifest["validation"]["split_counts"]["train"]

    test_lines = (first_dir / "test.jsonl").read_text(encoding="utf-8").splitlines()
    first_record = json.loads(test_lines[0])
    assert first_record["split"] == "test"
    assert set(first_record) >= {
        "prompt",
        "completion",
        "example_id",
        "pair_id",
        "case_role",
    }
