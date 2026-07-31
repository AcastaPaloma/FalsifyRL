from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

from dotenv import load_dotenv

from falsifyrl.autoscientist import (
    WorkflowState,
    create_client,
    run_autoscientist,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Launch a provenance-preserving AutoScientist refinement on the "
            "already audited FalsifyRL training dataset."
        )
    )
    parser.add_argument(
        "--state",
        type=Path,
        default=Path("outputs/autoscientist/workflow.json"),
    )
    parser.add_argument(
        "--model",
        default="meta-llama/Llama-3.2-3B-Instruct",
    )
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--epochs", type=float, default=3.0)
    parser.add_argument("--max-iterations", type=int, default=3)
    parser.add_argument("--target-win-rate", type=float, default=0.75)
    parser.add_argument("--timeout-seconds", type=float, default=43_200.0)
    return parser.parse_args()


def main() -> None:
    load_dotenv()
    args = parse_args()
    state = WorkflowState.load(args.state)
    if (
        state.autoscientist_status != "succeeded"
        or not state.autoscientist_run_id
        or not state.download_available
    ):
        raise ValueError("refinement requires a completed downloadable prior run")
    state.plan = replace(
        state.plan,
        model=args.model,
        max_iterations=args.max_iterations,
        target_win_rate=args.target_win_rate,
        hyperparams={
            "learning_rate": args.learning_rate,
            "n_epochs": args.epochs,
        },
    )
    state.save(args.state)
    state = run_autoscientist(
        create_client(),
        state,
        timeout=args.timeout_seconds,
        on_run_started=lambda current: current.save(args.state),
    )
    state.save(args.state)
    print(json.dumps(state.to_dict(), sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
