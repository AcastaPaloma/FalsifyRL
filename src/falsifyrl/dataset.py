from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from falsifyrl.scenarios import (
    GENERATOR_VERSION,
    SCENARIO_DEFINITIONS,
    GeneratedCase,
    generate_paired_cases,
)
from falsifyrl.schema import Diagnosis, ScenarioSplit
from falsifyrl.verifier import verify_case

FORBIDDEN_PROMPT_MARKERS = frozenset(
    {
        "gold_label",
        "injected_defect",
        "expected_verdict",
        "true_success",
        "validator_result",
        "generator_seed",
    }
)


@dataclass(frozen=True)
class DatasetBuildConfig:
    train_seed_count: int = 80
    validation_seed_count: int = 40
    test_seed_count: int = 40
    train_seed_offset: int = 0
    validation_seed_offset: int = 10_000
    test_seed_offset: int = 20_000

    def __post_init__(self) -> None:
        if min(
            self.train_seed_count,
            self.validation_seed_count,
            self.test_seed_count,
        ) <= 0:
            raise ValueError("all split seed counts must be positive")

    def seeds_for(self, split: ScenarioSplit) -> range:
        count_and_offset = {
            ScenarioSplit.TRAIN: (self.train_seed_count, self.train_seed_offset),
            ScenarioSplit.VALIDATION: (
                self.validation_seed_count,
                self.validation_seed_offset,
            ),
            ScenarioSplit.TEST: (self.test_seed_count, self.test_seed_offset),
        }
        count, offset = count_and_offset[split]
        return range(offset, offset + count)


def build_cases(config: DatasetBuildConfig) -> tuple[GeneratedCase, ...]:
    cases: list[GeneratedCase] = []
    for scenario in SCENARIO_DEFINITIONS:
        cases.extend(
            generate_paired_cases(
                seeds=config.seeds_for(scenario.split),
                scenarios=(scenario,),
            )
        )
    return tuple(cases)


def validate_cases(cases: tuple[GeneratedCase, ...]) -> dict[str, Any]:
    if not cases:
        raise ValueError("dataset must contain at least one case")

    ids = [case.example_id for case in cases]
    if len(ids) != len(set(ids)):
        raise ValueError("dataset contains duplicate example IDs")

    families_by_split: dict[str, set[str]] = defaultdict(set)
    splits_by_family: dict[str, set[str]] = defaultdict(set)
    pair_roles: dict[str, list[GeneratedCase]] = defaultdict(list)
    errors: list[str] = []
    for case in cases:
        split = case.scenario.split.value
        families_by_split[split].add(case.scenario.family)
        splits_by_family[case.scenario.family].add(split)
        pair_roles[case.pair_id].append(case)

        verification = verify_case(case)
        if not verification.valid:
            errors.append(f"{case.example_id}: {'; '.join(verification.errors)}")
        Diagnosis.from_json(case.diagnosis.to_json())

        prompt = case.render_prompt().lower()
        leaked = sorted(marker for marker in FORBIDDEN_PROMPT_MARKERS if marker in prompt)
        if leaked:
            errors.append(f"{case.example_id}: leaked prompt markers {leaked}")
        if case.example_id.lower() in prompt:
            errors.append(f"{case.example_id}: example ID leaked into prompt")

    overlapping_families = {
        family: sorted(splits)
        for family, splits in splits_by_family.items()
        if len(splits) != 1
    }
    if overlapping_families:
        errors.append(f"scenario families cross splits: {overlapping_families}")

    for pair_id, pair in pair_roles.items():
        roles = {case.case_role for case in pair}
        if len(pair) != 2 or roles != {"control", "exploit"}:
            errors.append(f"{pair_id}: expected one control and one exploit")
            continue
        control = next(case for case in pair if case.case_role == "control")
        exploit = next(case for case in pair if case.case_role == "exploit")
        if control.reward_spec != exploit.reward_spec:
            errors.append(f"{pair_id}: paired cases do not share the same reward program")
        if control.scenario != exploit.scenario or control.seed != exploit.seed:
            errors.append(f"{pair_id}: paired cases do not share scenario and seed")

    if errors:
        preview = "\n".join(errors[:20])
        raise ValueError(f"dataset validation failed with {len(errors)} error(s):\n{preview}")

    split_counts = Counter(case.scenario.split.value for case in cases)
    verdict_counts = Counter(case.diagnosis.verdict.value for case in cases)
    failure_counts = Counter(case.failure_type.value for case in cases)
    return {
        "case_count": len(cases),
        "pair_count": len(pair_roles),
        "split_counts": dict(sorted(split_counts.items())),
        "verdict_counts": dict(sorted(verdict_counts.items())),
        "failure_type_counts": dict(sorted(failure_counts.items())),
        "families_by_split": {
            split: sorted(families)
            for split, families in sorted(families_by_split.items())
        },
        "all_cases_verified": True,
        "all_pairs_reward_matched": True,
    }


def _record(case: GeneratedCase) -> dict[str, Any]:
    verification = verify_case(case)
    return {
        **case.training_record(),
        "failure_type": case.failure_type.value,
        "proxy_return": round(verification.proxy_return, 6),
        "aligned_proxy_return": round(verification.aligned_proxy_return, 6),
        "true_task_success": verification.true_success,
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_split(output_dir: Path, split: str, cases: list[GeneratedCase]) -> list[Path]:
    jsonl_path = output_dir / f"{split}.jsonl"
    csv_path = output_dir / f"{split}.csv"

    with jsonl_path.open("w", encoding="utf-8", newline="\n") as stream:
        for case in cases:
            stream.write(
                json.dumps(_record(case), sort_keys=True, separators=(",", ":")) + "\n"
            )

    with csv_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=("prompt", "completion"))
        writer.writeheader()
        for case in cases:
            writer.writerow(
                {
                    "prompt": case.render_prompt(),
                    "completion": case.diagnosis.to_json(),
                }
            )

    return [jsonl_path, csv_path]


def write_dataset(
    output_dir: str | Path,
    *,
    config: DatasetBuildConfig | None = None,
) -> dict[str, Any]:
    if config is None:
        config = DatasetBuildConfig()
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    cases = build_cases(config)
    validation = validate_cases(cases)
    files: list[Path] = []
    for split in ScenarioSplit:
        split_cases = [
            case for case in cases if case.scenario.split == split
        ]
        files.extend(_write_split(output_path, split.value, split_cases))

    file_manifest = {
        path.name: {
            "bytes": path.stat().st_size,
            "sha256": _sha256(path),
        }
        for path in sorted(files)
    }
    manifest: dict[str, Any] = {
        "dataset_name": "falsifyrl-seed",
        "dataset_version": "1.0.0",
        "generator_version": GENERATOR_VERSION,
        "config": asdict(config),
        "validation": validation,
        "files": file_manifest,
        "autoscientist_mapping": {
            "input_column": "prompt",
            "output_column": "completion",
            "recommended_training_file": "train.csv",
        },
    }
    manifest_path = output_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest
