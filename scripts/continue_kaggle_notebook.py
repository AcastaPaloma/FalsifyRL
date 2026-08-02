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


def load_selected_release_record(path: Path) -> dict:
    record = json.loads(path.read_text(encoding="utf-8"))
    if record.get("schema_version") != 1:
        raise ValueError("selected release record schema_version must be 1")
    identity = record.get("identity")
    kaggle_model = record.get("kaggle_model")
    kaggle_dataset = record.get("kaggle_dataset")
    huggingface_model = record.get("huggingface_model")
    if not all(
        isinstance(value, dict)
        for value in (identity, kaggle_model, kaggle_dataset, huggingface_model)
    ):
        raise ValueError("selected release record is missing required sections")
    required_identity = (
        "autoscientist_run_id",
        "base_model_id",
        "adapter_sha256",
        "base_predictions_sha256",
        "adapted_predictions_sha256",
    )
    if any(
        not isinstance(identity.get(key), str) or not identity[key]
        for key in required_identity
    ):
        raise ValueError("selected release record is missing identity evidence")
    if (
        not isinstance(kaggle_model.get("owner"), str)
        or not isinstance(kaggle_model.get("slug"), str)
        or kaggle_model.get("variation") != "lora"
        or not isinstance(kaggle_model.get("version"), int)
        or kaggle_model["version"] < 1
    ):
        raise ValueError("selected release record has an invalid Kaggle model source")
    expected_source = (
        f"{kaggle_model['owner']}/{kaggle_model['slug']}/pytorch/lora/"
        f"{kaggle_model['version']}"
    )
    if kaggle_model.get("source") != expected_source:
        raise ValueError("selected release record Kaggle model source is inconsistent")
    expected_model_url = (
        f"https://www.kaggle.com/models/{kaggle_model['owner']}/"
        f"{kaggle_model['slug']}/pytorch/lora"
    )
    if kaggle_model.get("url") != expected_model_url:
        raise ValueError("selected release record Kaggle model URL is inconsistent")
    expected_dataset_url = (
        f"https://www.kaggle.com/datasets/{kaggle_dataset.get('owner')}/"
        f"{kaggle_dataset.get('slug')}"
    )
    if (
        kaggle_dataset.get("owner") != kaggle_model["owner"]
        or kaggle_dataset.get("slug") != "falsifyrl-adapted"
        or kaggle_dataset.get("url") != expected_dataset_url
    ):
        raise ValueError("selected release record Kaggle dataset is inconsistent")
    expected_hf_url = (
        f"https://huggingface.co/{huggingface_model.get('owner')}/"
        f"{huggingface_model.get('slug')}"
    )
    if (
        not isinstance(huggingface_model.get("owner"), str)
        or not isinstance(huggingface_model.get("slug"), str)
        or huggingface_model.get("repo_id")
        != f"{huggingface_model['owner']}/{huggingface_model['slug']}"
        or huggingface_model.get("url") != expected_hf_url
    ):
        raise ValueError("selected release record Hugging Face model is inconsistent")
    return record


def resolve_kaggle_source(
    record: dict,
    *,
    owner: str | None,
    model_slug: str | None,
    model_version: int | None,
) -> tuple[str, str, int]:
    selected = record["kaggle_model"]
    selected_owner = selected["owner"]
    selected_slug = selected["slug"]
    selected_version = selected["version"]
    if owner is not None and owner != selected_owner:
        raise ValueError("Kaggle owner does not match the selected release record")
    if model_slug is not None and model_slug != selected_slug:
        raise ValueError("Kaggle model slug does not match the selected release record")
    if model_version is not None and model_version != selected_version:
        raise ValueError("Kaggle model version does not match the selected release record")
    return selected_owner, selected_slug, selected_version


def await_verified_model_release(
    manifest_path: Path,
    selected_release: dict,
    *,
    poll_seconds: float,
    timeout_seconds: float,
) -> dict:
    deadline = time.monotonic() + timeout_seconds
    while True:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (
            manifest["links"].get("kaggle_dataset")
            == selected_release["kaggle_dataset"]["url"]
            and manifest["links"].get("kaggle_model")
            == selected_release["kaggle_model"]["url"]
            and manifest["links"].get("huggingface_model")
            == selected_release["huggingface_model"]["url"]
            and manifest["attestations"].get("weights_public_on_both_platforms")
            is True
            and manifest["identifiers"].get("autoscientist_run_id")
            == selected_release["identity"]["autoscientist_run_id"]
            and manifest["identifiers"].get("base_model_id")
            == selected_release["identity"]["base_model_id"]
        ):
            return manifest
        if time.monotonic() >= deadline:
            raise TimeoutError("verified Kaggle model release did not become available")
        time.sleep(poll_seconds)


def _run(command: list[str], *, cwd: Path) -> subprocess.CompletedProcess[str]:
    print(json.dumps({"command": command}), flush=True)
    environment = os.environ.copy()
    environment["PYTHONUTF8"] = "1"
    environment["PYTHONIOENCODING"] = "utf-8"
    completed = subprocess.run(
        command,
        cwd=cwd,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
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


def verify_kaggle_evaluation_report(report: dict, selected_release: dict) -> None:
    if report.get("example_count") != 640:
        raise RuntimeError("Kaggle evaluation did not run all 640 held-out examples")
    if report.get("prediction_mode") != "commit_verified_colab_evidence":
        raise RuntimeError("public Kaggle run did not use the released evidence bundle")
    if report.get("release_identity") != selected_release["identity"]:
        raise RuntimeError("Kaggle report release identity does not match selected release")
    base = report["base_metrics"]
    adapted = report["adapted_metrics"]
    if adapted["json_validity"] < 0.95:
        raise RuntimeError("Kaggle adapter JSON validity is below 95%")
    if adapted["verdict_macro_f1"] <= base["verdict_macro_f1"]:
        raise RuntimeError("Kaggle adapter does not improve verdict macro-F1")


def parse_kernel_status(output: str) -> str:
    normalized = output.lower()
    enum_match = re.search(
        r"kernelworkerstatus[.]"
        r"(complete|running|queued|pending|error|failed|cancelled|canceled)",
        normalized,
    )
    if enum_match:
        return enum_match.group(1)
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
        "--selected-release-record",
        type=Path,
        required=True,
        help="write-once record created by continue_model_release.py",
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
    parser.add_argument("--model-slug")
    parser.add_argument("--model-version", type=int)
    parser.add_argument(
        "--kernel-slug",
        default="falsifyrl-held-out-reward-hacking-evaluation",
    )
    parser.add_argument(
        "--skip-push",
        action="store_true",
        help="resume polling an already-pushed notebook without creating a new version",
    )
    parser.add_argument("--poll-seconds", type=float, default=30.0)
    parser.add_argument("--timeout-seconds", type=float, default=259_200.0)
    return parser.parse_args()


def main() -> None:
    load_dotenv()
    args = parse_args()
    selected_release = load_selected_release_record(args.selected_release_record)
    owner, model_slug, model_version = resolve_kaggle_source(
        selected_release,
        owner=args.owner or os.environ.get("FALSIFYRL_KAGGLE_OWNER"),
        model_slug=args.model_slug,
        model_version=args.model_version,
    )
    if not args.kaggle_cli.is_file():
        raise FileNotFoundError(args.kaggle_cli)
    await_verified_model_release(
        args.submission_manifest,
        selected_release,
        poll_seconds=args.poll_seconds,
        timeout_seconds=args.timeout_seconds,
    )
    repository = Path(__file__).resolve().parents[1]
    kernel_handle = f"{owner}/{args.kernel_slug}"
    if not args.skip_push:
        _run(
            [
                sys.executable,
                "scripts/prepare_kaggle_notebook.py",
                "--owner",
                owner,
                "--model-version",
                str(model_version),
                "--model-slug",
                model_slug,
                "--notebook",
                str(args.notebook.resolve()),
                "--metadata-template",
                str(args.metadata_template.resolve()),
                "--output-dir",
                str(args.bundle_dir.resolve()),
            ],
            cwd=repository,
        )
        _run(
            [
                str(args.kaggle_cli.resolve()),
                "kernels",
                "push",
                "-p",
                str(args.bundle_dir.resolve()),
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
    verify_kaggle_evaluation_report(report, selected_release)
    base = report["base_metrics"]
    adapted = report["adapted_metrics"]

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
