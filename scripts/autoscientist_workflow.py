from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from falsifyrl.autoscientist import (
    AutoScientistPlan,
    WorkflowState,
    create_client,
    download_checkpoint,
    ingest_and_estimate,
    refresh_status,
    run_adaptation,
    run_autoscientist,
)


def _plan_from_args(args: argparse.Namespace) -> AutoScientistPlan:
    source_url = args.source_url
    if source_url is None and args.source == "huggingface":
        source_url = os.environ.get("FALSIFYRL_HF_DATASET_URL")
    if source_url is None and args.source == "kaggle":
        source_url = os.environ.get("FALSIFYRL_KAGGLE_DATASET_URL")
    return AutoScientistPlan(
        source=args.source,
        source_url=source_url,
        source_file=args.source_file,
        local_file=None if args.local_file is None else str(args.local_file),
        max_iterations=args.max_iterations,
        target_win_rate=args.target_win_rate,
        model=args.model,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Credential-safe staged AutoScientist workflow for FalsifyRL."
    )
    subparsers = parser.add_subparsers(dest="action", required=True)

    plan_parser = subparsers.add_parser("plan")
    plan_parser.add_argument(
        "--source",
        choices=("file", "huggingface", "kaggle"),
        default="huggingface",
    )
    plan_parser.add_argument("--source-url")
    plan_parser.add_argument("--source-file", default="train.csv")
    plan_parser.add_argument("--local-file", type=Path)
    plan_parser.add_argument("--max-iterations", type=int, default=3)
    plan_parser.add_argument("--target-win-rate", type=float, default=0.75)
    plan_parser.add_argument("--model")
    plan_parser.add_argument(
        "--state",
        type=Path,
        default=Path("outputs/autoscientist/workflow.json"),
    )

    for action in ("ingest", "adapt", "train", "status", "download"):
        action_parser = subparsers.add_parser(action)
        action_parser.add_argument(
            "--state",
            type=Path,
            default=Path("outputs/autoscientist/workflow.json"),
        )
        if action == "download":
            action_parser.add_argument(
                "--checkpoint",
                type=Path,
                default=Path("outputs/autoscientist/best-checkpoint.tgz"),
            )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.action == "plan":
        state = WorkflowState(plan=_plan_from_args(args))
        state.save(args.state)
    else:
        state = WorkflowState.load(args.state)
        client = create_client()
        if args.action == "ingest":
            state = ingest_and_estimate(client, state)
        elif args.action == "adapt":
            state = run_adaptation(client, state)
        elif args.action == "train":
            state = run_autoscientist(client, state)
        elif args.action == "status":
            state = refresh_status(client, state)
        elif args.action == "download":
            checkpoint = download_checkpoint(client, state, args.checkpoint)
            print(json.dumps({"checkpoint": str(checkpoint.resolve())}, indent=2))
        state.save(args.state)
    print(json.dumps(state.to_dict(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

