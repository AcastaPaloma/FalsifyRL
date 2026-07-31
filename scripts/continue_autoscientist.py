from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from falsifyrl.autoscientist import (
    WorkflowState,
    create_client,
    export_and_audit_adapted_dataset,
    prepare_training_dataset,
    run_autoscientist,
)


def _value(record: Any, key: str, default: Any = None) -> Any:
    if isinstance(record, dict):
        return record.get(key, default)
    return getattr(record, key, default)


def await_existing_adaptation(
    client: Any,
    state: WorkflowState,
    state_path: Path,
    *,
    poll_seconds: float,
    timeout_seconds: float,
) -> WorkflowState:
    """Wait for an existing dataset run without creating another paid run."""
    if not state.dataset_id or not state.adaptation_run_id:
        raise ValueError("workflow state has no existing adaptation run")

    deadline = time.monotonic() + timeout_seconds
    last_processed: int | None = None
    while True:
        record = client.datasets.get(state.dataset_id)
        status = str(_value(record, "status"))
        progress = _value(record, "progress", {}) or {}
        processed = int(_value(progress, "processed_rows", 0) or 0)
        total = int(
            _value(progress, "total_rows", state.plan.expected_training_rows)
            or state.plan.expected_training_rows
        )
        if processed != last_processed or status != "running":
            print(
                json.dumps(
                    {
                        "dataset_id": state.dataset_id,
                        "processed_rows": processed,
                        "status": status,
                        "total_rows": total,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
            last_processed = processed

        if status == "succeeded":
            state.dataset_status = status
            state.adapted_schema_valid = False
            state.save(state_path)
            return state
        if status == "failed":
            error_data = _value(record, "error_data")
            raise RuntimeError(f"dataset adaptation failed: {error_data}")
        if time.monotonic() >= deadline:
            raise TimeoutError(
                f"existing dataset adaptation timed out with status {status}"
            )
        time.sleep(poll_seconds)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Resume the existing FalsifyRL adaptation, audit its exact export, "
            "and launch AutoScientist without creating another dataset run."
        )
    )
    parser.add_argument(
        "--state",
        type=Path,
        default=Path("outputs/autoscientist/workflow.json"),
    )
    parser.add_argument(
        "--adapted-csv",
        type=Path,
        default=Path("outputs/autoscientist/adapted-train.csv"),
    )
    parser.add_argument(
        "--source-train-csv",
        type=Path,
        default=Path("outputs/falsifyrl_seed_v1/train.csv"),
    )
    parser.add_argument("--poll-seconds", type=float, default=30.0)
    parser.add_argument("--adapt-timeout-seconds", type=float, default=21_600.0)
    return parser.parse_args()


def main() -> None:
    load_dotenv()
    args = parse_args()
    state = WorkflowState.load(args.state)
    client = create_client()

    state = await_existing_adaptation(
        client,
        state,
        args.state,
        poll_seconds=args.poll_seconds,
        timeout_seconds=args.adapt_timeout_seconds,
    )
    if not state.plan.model:
        raise ValueError("workflow state must pin an explicit training model")

    state = export_and_audit_adapted_dataset(
        client,
        state,
        args.adapted_csv,
        args.source_train_csv,
    )
    state.save(args.state)
    print(
        json.dumps(
            {
                "adapted_export_path": state.adapted_export_path,
                "adapted_export_sha256": state.adapted_export_sha256,
                "adapted_row_count": state.adapted_row_count,
                "adapted_schema_valid": state.adapted_schema_valid,
            },
            sort_keys=True,
        ),
        flush=True,
    )

    state = prepare_training_dataset(
        client,
        state,
        on_dataset_created=lambda current: current.save(args.state),
    )
    state.save(args.state)
    print(
        json.dumps(
            {
                "training_dataset_id": state.training_dataset_id,
                "training_dataset_status": state.training_dataset_status,
                "training_prompt_column": state.training_prompt_column,
                "training_completion_column": state.training_completion_column,
            },
            sort_keys=True,
        ),
        flush=True,
    )

    state = run_autoscientist(
        client,
        state,
        timeout=43_200,
        on_run_started=lambda current: current.save(args.state),
    )
    state.save(args.state)
    print(json.dumps(state.to_dict(), sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
