from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from pathlib import Path

from dotenv import load_dotenv

from falsifyrl.autoscientist import (
    WorkflowState,
    create_client,
    download_checkpoint,
)
from falsifyrl.release import extract_adapter_checkpoint


def await_successful_training(
    state_path: Path,
    *,
    poll_seconds: float,
    timeout_seconds: float,
) -> WorkflowState:
    deadline = time.monotonic() + timeout_seconds
    while True:
        state = WorkflowState.load(state_path)
        if (
            state.autoscientist_status == "succeeded"
            and state.autoscientist_run_id
            and state.download_available
            and state.best_win_rate is not None
        ):
            return state
        if state.autoscientist_status == "failed":
            raise RuntimeError("AutoScientist training failed")
        if time.monotonic() >= deadline:
            raise TimeoutError("successful AutoScientist checkpoint did not become available")
        time.sleep(poll_seconds)


def locate_or_extract_adapter(checkpoint: Path, destination: Path) -> Path:
    if destination.exists() and any(destination.iterdir()):
        adapter_configs = list(destination.rglob("adapter_config.json"))
        if len(adapter_configs) != 1:
            raise ValueError(
                "existing extraction must contain exactly one adapter_config.json"
            )
        adapter_root = adapter_configs[0].parent
        if not (adapter_root / "adapter_model.safetensors").is_file():
            raise ValueError("existing extraction is missing adapter_model.safetensors")
        return adapter_root
    return extract_adapter_checkpoint(checkpoint, destination)


def _subprocess_environment(repository: Path) -> dict[str, str]:
    environment = os.environ.copy()
    python_paths = [str((repository / "src").resolve())]
    if existing := environment.get("PYTHONPATH"):
        python_paths.append(existing)
    environment["PYTHONPATH"] = os.pathsep.join(python_paths)
    return environment


def _run(command: list[str], *, cwd: Path) -> None:
    print(json.dumps({"command": command}), flush=True)
    subprocess.run(
        command,
        cwd=cwd,
        check=True,
        env=_subprocess_environment(cwd),
    )


def update_private_manifest(
    manifest_path: Path,
    state: WorkflowState,
    trained_report: dict,
) -> None:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["identifiers"]["autoscientist_run_id"] = state.autoscientist_run_id
    manifest["metrics"]["autoscientist_best_win_rate"] = state.best_win_rate
    manifest["metrics"]["trained_model_composite"] = trained_report["metrics"][
        "composite_score"
    ]
    manifest["metrics"]["trained_json_validity"] = trained_report["metrics"][
        "json_validity"
    ]
    temporary = manifest_path.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(manifest_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Wait for the best AutoScientist checkpoint, securely extract it, "
            "and run the exact held-out adapter evaluation."
        )
    )
    parser.add_argument(
        "--state",
        type=Path,
        default=Path("outputs/autoscientist/workflow.json"),
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("outputs/autoscientist/best-checkpoint.tgz"),
    )
    parser.add_argument(
        "--adapter-dir",
        type=Path,
        default=Path("outputs/autoscientist/extracted-checkpoint"),
    )
    parser.add_argument(
        "--inference-python",
        type=Path,
        default=Path("outputs/inference-venv/Scripts/python.exe"),
    )
    parser.add_argument(
        "--predictions",
        type=Path,
        default=Path("outputs/evaluation/model-test-predictions.jsonl"),
    )
    parser.add_argument(
        "--trained-report",
        type=Path,
        default=Path("outputs/evaluation/model-test.json"),
    )
    parser.add_argument(
        "--base-report",
        type=Path,
        default=Path("outputs/evaluation/base-test.json"),
    )
    parser.add_argument(
        "--dataset-manifest",
        type=Path,
        default=Path("artifacts/release/adapted-dataset/release-manifest.json"),
    )
    parser.add_argument(
        "--comparison-json",
        type=Path,
        default=Path("outputs/evaluation/comparison.json"),
    )
    parser.add_argument(
        "--comparison-markdown",
        type=Path,
        default=Path("outputs/evaluation/comparison.md"),
    )
    parser.add_argument(
        "--submission-manifest",
        type=Path,
        default=Path("outputs/submission/manifest.json"),
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--poll-seconds", type=float, default=30.0)
    parser.add_argument("--timeout-seconds", type=float, default=86_400.0)
    return parser.parse_args()


def main() -> None:
    load_dotenv()
    args = parse_args()
    repository = Path(__file__).resolve().parents[1]
    state = await_successful_training(
        args.state,
        poll_seconds=args.poll_seconds,
        timeout_seconds=args.timeout_seconds,
    )
    if not args.inference_python.is_file():
        raise FileNotFoundError(args.inference_python)

    if not args.checkpoint.is_file():
        checkpoint = download_checkpoint(create_client(), state, args.checkpoint)
    else:
        checkpoint = args.checkpoint
    adapter_root = locate_or_extract_adapter(checkpoint, args.adapter_dir)
    adapter_config = json.loads(
        (adapter_root / "adapter_config.json").read_text(encoding="utf-8")
    )
    base_model_id = str(adapter_config["base_model_name_or_path"])

    _run(
        [
            str(args.inference_python.resolve()),
            "scripts/generate_model_predictions.py",
            "--model-id",
            base_model_id,
            "--adapter",
            str(adapter_root.resolve()),
            "--split",
            "test",
            "--max-new-tokens",
            "256",
            "--batch-size",
            str(args.batch_size),
            "--resume",
            "--output",
            str(args.predictions.resolve()),
        ],
        cwd=repository,
    )
    _run(
        [
            str(Path(sys_executable()).resolve()),
            "scripts/evaluate_baselines.py",
            "--split",
            "test",
            "--predictions",
            str(args.predictions.resolve()),
            "--output",
            str(args.trained_report.resolve()),
        ],
        cwd=repository,
    )

    trained_report = json.loads(args.trained_report.read_text(encoding="utf-8"))
    base_report = json.loads(args.base_report.read_text(encoding="utf-8"))
    trained_metrics = trained_report["metrics"]
    if trained_metrics["composite_score"] <= base_report["metrics"]["composite_score"]:
        raise RuntimeError("trained adapter did not improve held-out composite score")
    if trained_metrics["json_validity"] < 0.95:
        raise RuntimeError("trained adapter JSON validity is below 95%")
    if state.best_win_rate is None or state.best_win_rate <= 0.5:
        raise RuntimeError("AutoScientist best win rate does not exceed 0.5")

    deadline = time.monotonic() + args.timeout_seconds
    while not args.dataset_manifest.is_file():
        if time.monotonic() >= deadline:
            raise TimeoutError("adapted dataset release manifest did not become available")
        time.sleep(args.poll_seconds)

    _run(
        [
            str(Path(sys_executable()).resolve()),
            "scripts/build_evaluation_report.py",
            "--base-report",
            str(args.base_report.resolve()),
            "--adapted-report",
            str(args.trained_report.resolve()),
            "--base-model-id",
            base_model_id,
            "--dataset-manifest",
            str(args.dataset_manifest.resolve()),
            "--adapter",
            str((adapter_root / "adapter_model.safetensors").resolve()),
            "--autoscientist-run-id",
            str(state.autoscientist_run_id),
            "--json-output",
            str(args.comparison_json.resolve()),
            "--markdown-output",
            str(args.comparison_markdown.resolve()),
        ],
        cwd=repository,
    )
    update_private_manifest(args.submission_manifest, state, trained_report)
    print(
        json.dumps(
            {
                "adapter_root": str(adapter_root.resolve()),
                "base_model_id": base_model_id,
                "comparison_json": str(args.comparison_json.resolve()),
                "trained_metrics": trained_metrics,
            },
            indent=2,
            sort_keys=True,
        ),
        flush=True,
    )


def sys_executable() -> str:
    import sys

    return sys.executable


if __name__ == "__main__":
    main()
