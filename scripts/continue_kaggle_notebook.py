from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

from dotenv import load_dotenv


def await_verified_model_release(
    manifest_path: Path,
    *,
    poll_seconds: float,
    timeout_seconds: float,
) -> dict:
    deadline = time.monotonic() + timeout_seconds
    while True:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (
            manifest["links"].get("kaggle_dataset")
            and manifest["links"].get("kaggle_model")
            and manifest["attestations"].get("weights_public_on_both_platforms")
            is True
        ):
            return manifest
        if time.monotonic() >= deadline:
            raise TimeoutError("verified Kaggle model release did not become available")
        time.sleep(poll_seconds)


def _run(command: list[str], *, cwd: Path) -> subprocess.CompletedProcess[str]:
    print(json.dumps({"command": command}), flush=True)
    completed = subprocess.run(
        command,
        cwd=cwd,
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.stdout:
        print(completed.stdout, flush=True)
    if completed.stderr:
        print(completed.stderr, flush=True)
    if completed.returncode != 0:
        raise subprocess.CalledProcessError(
            completed.returncode,
            command,
            completed.stdout,
            completed.stderr,
        )
    return completed


def update_private_manifest(manifest_path: Path, notebook_url: str) -> None:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["links"]["kaggle_notebook"] = notebook_url
    temporary = manifest_path.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(manifest_path)


def parse_kernel_status(output: str) -> str:
    normalized = output.lower()
    match = re.search(
        r"\b(?:kernel\s+)?status\s*[:=]?\s*[\"']?"
        r"(complete|running|queued|pending|error|failed|cancelled|canceled)",
        normalized,
    )
    if match:
        return match.group(1)
    for candidate in (
        "complete",
        "running",
        "queued",
        "pending",
        "error",
        "failed",
        "cancelled",
        "canceled",
    ):
        if normalized.strip() == candidate:
            return candidate
    return "unknown"


def await_kernel_completion(
    *,
    kaggle_cli: Path,
    kernel_handle: str,
    repository: Path,
    poll_seconds: float,
    timeout_seconds: float,
) -> subprocess.CompletedProcess[str]:
    deadline = time.monotonic() + timeout_seconds
    while True:
        completed = _run(
            [
                str(kaggle_cli.resolve()),
                "kernels",
                "status",
                kernel_handle,
            ],
            cwd=repository,
        )
        status = parse_kernel_status(completed.stdout + "\n" + completed.stderr)
        if status == "complete":
            return completed
        if status in {"error", "failed", "cancelled", "canceled"}:
            raise RuntimeError(
                f"Kaggle notebook ended with terminal status {status}: "
                f"{completed.stdout.strip()}"
            )
        if time.monotonic() >= deadline:
            raise TimeoutError(
                f"Kaggle notebook timed out with status {status}: "
                f"{completed.stdout.strip()}"
            )
        time.sleep(poll_seconds)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Publish the exact Kaggle held-out notebook after verified model "
            "release, wait for its run, and audit its downloaded report."
        )
    )
    parser.add_argument(
        "--submission-manifest",
        type=Path,
        default=Path("outputs/submission/manifest.json"),
    )
    parser.add_argument(
        "--notebook",
        type=Path,
        default=Path("kaggle/falsifyrl_evaluation.ipynb"),
    )
    parser.add_argument(
        "--metadata-template",
        type=Path,
        default=Path("kaggle/kernel-metadata.template.json"),
    )
    parser.add_argument(
        "--bundle-dir",
        type=Path,
        default=Path("artifacts/release/kaggle-notebook"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/kaggle-evaluation"),
    )
    parser.add_argument(
        "--kaggle-cli",
        type=Path,
        default=Path("outputs/kaggle-cli-venv/Scripts/kaggle.exe"),
    )
    parser.add_argument("--owner")
    parser.add_argument("--model-slug", default="falsifyrl-autoscientist")
    parser.add_argument("--model-version", type=int, default=1)
    parser.add_argument("--poll-seconds", type=float, default=30.0)
    parser.add_argument("--timeout-seconds", type=float, default=259_200.0)
    return parser.parse_args()


def main() -> None:
    load_dotenv()
    args = parse_args()
    owner = args.owner or os.environ.get("FALSIFYRL_KAGGLE_OWNER")
    if not owner:
        raise RuntimeError("Kaggle owner is required")
    if not args.kaggle_cli.is_file():
        raise FileNotFoundError(args.kaggle_cli)
    await_verified_model_release(
        args.submission_manifest,
        poll_seconds=args.poll_seconds,
        timeout_seconds=args.timeout_seconds,
    )
    repository = Path(__file__).resolve().parents[1]
    _run(
        [
            sys.executable,
            "scripts/prepare_kaggle_notebook.py",
            "--owner",
            owner,
            "--model-version",
            str(args.model_version),
            "--model-slug",
            args.model_slug,
            "--notebook",
            str(args.notebook.resolve()),
            "--metadata-template",
            str(args.metadata_template.resolve()),
            "--output-dir",
            str(args.bundle_dir.resolve()),
        ],
        cwd=repository,
    )
    kernel_handle = f"{owner}/falsifyrl-held-out-evaluation"
    _run(
        [
            str(args.kaggle_cli.resolve()),
            "kernels",
            "push",
            "-p",
            str(args.bundle_dir.resolve()),
            "--accelerator",
            "NvidiaTeslaP100",
            "--timeout",
            "7200",
        ],
        cwd=repository,
    )
    await_kernel_completion(
        kaggle_cli=args.kaggle_cli,
        kernel_handle=kernel_handle,
        repository=repository,
        poll_seconds=args.poll_seconds,
        timeout_seconds=args.timeout_seconds,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    _run(
        [
            str(args.kaggle_cli.resolve()),
            "kernels",
            "output",
            kernel_handle,
            "-p",
            str(args.output_dir.resolve()),
            "--force",
        ],
        cwd=repository,
    )
    report_path = args.output_dir / "kaggle-evaluation.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("example_count") != 640:
        raise RuntimeError("Kaggle evaluation did not run all 640 held-out examples")
    base = report["base_metrics"]
    adapted = report["adapted_metrics"]
    if adapted["json_validity"] < 0.95:
        raise RuntimeError("Kaggle adapter JSON validity is below 95%")
    if adapted["verdict_macro_f1"] <= base["verdict_macro_f1"]:
        raise RuntimeError("Kaggle adapter does not improve verdict macro-F1")

    notebook_url = f"https://www.kaggle.com/code/{kernel_handle}"
    update_private_manifest(args.submission_manifest, notebook_url)
    print(
        json.dumps(
            {
                "notebook_url": notebook_url,
                "report": str(report_path.resolve()),
                "example_count": report["example_count"],
                "base_verdict_macro_f1": base["verdict_macro_f1"],
                "adapted_verdict_macro_f1": adapted["verdict_macro_f1"],
                "adapted_json_validity": adapted["json_validity"],
            },
            indent=2,
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
