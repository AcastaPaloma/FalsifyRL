from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
import time
from pathlib import Path

from dotenv import load_dotenv

from falsifyrl.autoscientist import WorkflowState
from falsifyrl.release import (
    prepare_adapted_dataset_bundle,
    publish_huggingface_dataset,
    publish_kaggle_dataset,
    verify_anonymous_public_page,
)


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def await_audited_export(
    state_path: Path,
    *,
    poll_seconds: float,
    timeout_seconds: float,
) -> WorkflowState:
    deadline = time.monotonic() + timeout_seconds
    while True:
        state = WorkflowState.load(state_path)
        if (
            state.dataset_status == "succeeded"
            and state.adapted_schema_valid
            and state.adapted_row_count
            and state.adapted_row_count <= state.plan.expected_training_rows
            and state.adapted_export_path
            and state.adapted_audit_path
            and state.adapted_export_sha256
        ):
            return state
        if time.monotonic() >= deadline:
            raise TimeoutError("audited adapted export did not become available")
        time.sleep(poll_seconds)


def verify_huggingface_training_file(repo_id: str, expected_sha256: str) -> str:
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as error:
        raise RuntimeError(
            "Install release dependencies with `pip install -e .[release]`."
        ) from error
    downloaded = hf_hub_download(
        repo_id=repo_id,
        filename="train.csv",
        repo_type="dataset",
        token=False,
        force_download=True,
    )
    actual = _sha256(downloaded)
    if actual != expected_sha256:
        raise ValueError(
            f"Hugging Face train.csv hash mismatch: {actual} != {expected_sha256}"
        )
    return actual


def verify_kaggle_training_file(handle: str, expected_sha256: str) -> str:
    try:
        import kagglehub
    except ImportError as error:
        raise RuntimeError(
            "Install release dependencies with `pip install -e .[release]`."
        ) from error
    with tempfile.TemporaryDirectory(prefix="falsifyrl-kaggle-verify-") as temporary:
        downloaded = Path(
            kagglehub.dataset_download(
                handle,
                path="train.csv",
                force_download=True,
                output_dir=temporary,
            )
        )
        if downloaded.is_dir():
            downloaded = downloaded / "train.csv"
        actual = _sha256(downloaded)
    if actual != expected_sha256:
        raise ValueError(f"Kaggle train.csv hash mismatch: {actual} != {expected_sha256}")
    return actual


def update_private_manifest(
    manifest_path: Path,
    *,
    huggingface_url: str,
    kaggle_url: str,
    manifest_url: str,
) -> None:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["links"]["huggingface_dataset"] = huggingface_url
    manifest["links"]["kaggle_dataset"] = kaggle_url
    manifest["dataset"]["variant"] = "adapted"
    manifest["dataset"]["sha256_manifest"] = manifest_url
    manifest["attestations"]["dataset_public_on_both_platforms"] = True
    manifest["attestations"]["same_dataset_used_for_training"] = True
    temporary = manifest_path.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(manifest_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Wait for the exact adapted export, publish it to both dataset hosts, "
            "and verify both public train.csv hashes."
        )
    )
    parser.add_argument(
        "--state",
        type=Path,
        default=Path("outputs/autoscientist/workflow.json"),
    )
    parser.add_argument(
        "--seed-dataset-dir",
        type=Path,
        default=Path("outputs/falsifyrl_seed_v1"),
    )
    parser.add_argument(
        "--bundle-dir",
        type=Path,
        default=Path("artifacts/release/adapted-dataset"),
    )
    parser.add_argument(
        "--submission-manifest",
        type=Path,
        default=Path("outputs/submission/manifest.json"),
    )
    parser.add_argument("--huggingface-owner")
    parser.add_argument("--kaggle-owner")
    parser.add_argument("--slug", default="falsifyrl-adapted")
    parser.add_argument("--poll-seconds", type=float, default=30.0)
    parser.add_argument("--timeout-seconds", type=float, default=43_200.0)
    return parser.parse_args()


def main() -> None:
    load_dotenv()
    args = parse_args()
    huggingface_owner = args.huggingface_owner or os.environ.get("FALSIFYRL_HF_OWNER")
    kaggle_owner = args.kaggle_owner or os.environ.get("FALSIFYRL_KAGGLE_OWNER")
    if not huggingface_owner or not kaggle_owner:
        raise RuntimeError("both Hugging Face and Kaggle owners are required")

    state = await_audited_export(
        args.state,
        poll_seconds=args.poll_seconds,
        timeout_seconds=args.timeout_seconds,
    )
    adapted_path = Path(state.adapted_export_path)
    audit_path = Path(state.adapted_audit_path)
    release_manifest = prepare_adapted_dataset_bundle(
        args.seed_dataset_dir,
        adapted_path,
        audit_path,
        args.bundle_dir,
    )
    expected_sha256 = str(release_manifest["training_file_sha256"])
    if expected_sha256 != state.adapted_export_sha256:
        raise ValueError("release manifest does not match the audited workflow export")

    huggingface_url = publish_huggingface_dataset(
        args.bundle_dir,
        owner=huggingface_owner,
        slug=args.slug,
    )
    kaggle_url = publish_kaggle_dataset(
        args.bundle_dir,
        owner=kaggle_owner,
        slug=args.slug,
    )
    verify_anonymous_public_page(
        kaggle_url,
        expected_marker=args.slug,
    )
    huggingface_repo_id = f"{huggingface_owner}/{args.slug}"
    kaggle_handle = f"{kaggle_owner}/{args.slug}"
    verification = {
        "expected_sha256": expected_sha256,
        "huggingface_sha256": verify_huggingface_training_file(
            huggingface_repo_id,
            expected_sha256,
        ),
        "kaggle_sha256": verify_kaggle_training_file(
            kaggle_handle,
            expected_sha256,
        ),
        "huggingface_url": huggingface_url,
        "kaggle_url": kaggle_url,
    }
    verification_path = args.bundle_dir / "publication-verification.json"
    verification_path.write_text(
        json.dumps(verification, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    manifest_url = (
        f"{huggingface_url}/resolve/main/release-manifest.json"
    )
    update_private_manifest(
        args.submission_manifest,
        huggingface_url=huggingface_url,
        kaggle_url=kaggle_url,
        manifest_url=manifest_url,
    )
    print(json.dumps(verification, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
