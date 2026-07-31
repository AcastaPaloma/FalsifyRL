from __future__ import annotations

import argparse
import json
from pathlib import Path

from falsifyrl.autoscientist import WorkflowState
from falsifyrl.release import prepare_model_bundle


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Securely extract and audit the best AutoScientist checkpoint."
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("outputs/autoscientist/best-checkpoint.tgz"),
    )
    parser.add_argument(
        "--workflow-state",
        type=Path,
        default=Path("outputs/autoscientist/workflow.json"),
    )
    parser.add_argument("--dataset-repo-id", required=True)
    parser.add_argument(
        "--evaluation-report",
        type=Path,
        default=Path("outputs/evaluation/model-test.json"),
    )
    parser.add_argument(
        "--bundle-dir",
        type=Path,
        default=Path("artifacts/release/model"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    state = WorkflowState.load(args.workflow_state)
    missing = [
        name
        for name, value in {
            "autoscientist_run_id": state.autoscientist_run_id,
            "resolved_model": state.resolved_model,
            "best_win_rate": state.best_win_rate,
        }.items()
        if value is None
    ]
    if missing:
        raise RuntimeError(f"workflow state is missing final values: {missing}")
    manifest = prepare_model_bundle(
        args.checkpoint,
        args.bundle_dir,
        base_model_id=state.resolved_model,
        dataset_repo_id=args.dataset_repo_id,
        autoscientist_run_id=state.autoscientist_run_id,
        best_win_rate=state.best_win_rate,
        evaluation_report=args.evaluation_report,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

