from __future__ import annotations

import argparse
import json
from pathlib import Path

from falsifyrl.dataset import DatasetBuildConfig, build_cases
from falsifyrl.evaluation import evaluate_completions, load_prediction_jsonl
from falsifyrl.schema import Diagnosis, FailureType, Verdict


def _always_aligned_completion() -> str:
    return Diagnosis(
        verdict=Verdict.ALIGNED,
        failure_type=FailureType.NONE,
        responsible_agents=(),
        evidence_steps=(),
        counterexample_config={},
        reward_patch=None,
        expected_effect="No patch is needed.",
        confidence=0.5,
    ).to_json()


def _baseline_completions(cases: list, baseline: str) -> dict[str, str]:
    if baseline == "oracle-ceiling":
        return {case.example_id: case.diagnosis.to_json() for case in cases}
    if baseline == "always-aligned":
        completion = _always_aligned_completion()
        return {case.example_id: completion for case in cases}
    if baseline == "reward-only":
        exploits_by_pair = {
            case.pair_id: case
            for case in cases
            if case.case_role == "exploit"
        }
        return {
            case.example_id: exploits_by_pair[case.pair_id].diagnosis.to_json()
            for case in cases
        }
    raise ValueError(f"unknown baseline: {baseline}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate FalsifyRL baselines or model outputs.")
    parser.add_argument(
        "--split",
        choices=("validation", "test"),
        default="test",
    )
    parser.add_argument(
        "--baseline",
        choices=("always-aligned", "reward-only", "oracle-ceiling"),
    )
    parser.add_argument("--predictions", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if (args.baseline is None) == (args.predictions is None):
        parser.error("provide exactly one of --baseline or --predictions")
    return args


def main() -> None:
    args = parse_args()
    cases = [
        case
        for case in build_cases(DatasetBuildConfig())
        if case.scenario.split.value == args.split
    ]
    completions = (
        _baseline_completions(cases, args.baseline)
        if args.baseline
        else load_prediction_jsonl(args.predictions)
    )
    result = {
        "split": args.split,
        "source": args.baseline or str(args.predictions),
        "metrics": evaluate_completions(cases, completions).to_dict(),
    }
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()

