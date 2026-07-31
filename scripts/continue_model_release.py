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
from falsifyrl.demo import prepare_space_bundle
from falsifyrl.release import (
    prepare_model_bundle,
    publish_huggingface_model,
    publish_huggingface_space,
    publish_kaggle_model,
)


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def await_passing_evaluation(
    state_path: Path,
    comparison_path: Path,
    submission_manifest_path: Path,
    *,
    poll_seconds: float,
    timeout_seconds: float,
) -> tuple[WorkflowState, dict, dict]:
    deadline = time.monotonic() + timeout_seconds
    while True:
        state = WorkflowState.load(state_path)
        submission = json.loads(
            submission_manifest_path.read_text(encoding="utf-8")
        )
        if (
            state.autoscientist_status == "succeeded"
            and state.autoscientist_run_id
            and state.best_win_rate is not None
            and comparison_path.is_file()
            and submission["links"].get("huggingface_dataset")
            and submission["links"].get("kaggle_dataset")
        ):
            comparison = json.loads(comparison_path.read_text(encoding="utf-8"))
            trained = comparison["metrics"]["adapted"]
            base = comparison["metrics"]["base"]
            if trained["composite_score"] <= base["composite_score"]:
                raise RuntimeError("comparison does not prove held-out improvement")
            if trained["json_validity"] < 0.95:
                raise RuntimeError("comparison JSON validity is below 95%")
            if state.best_win_rate <= 0.5:
                raise RuntimeError("AutoScientist best win rate does not exceed 0.5")
            return state, comparison, submission
        if time.monotonic() >= deadline:
            raise TimeoutError("passing held-out evaluation did not become available")
        time.sleep(poll_seconds)


def verify_huggingface_adapter(repo_id: str, expected_sha256: str) -> str:
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as error:
        raise RuntimeError(
            "Install release dependencies with `pip install -e .[release]`."
        ) from error
    downloaded = hf_hub_download(
        repo_id=repo_id,
        filename="adapter_model.safetensors",
        token=False,
        force_download=True,
    )
    actual = _sha256(downloaded)
    if actual != expected_sha256:
        raise ValueError(
            f"Hugging Face adapter hash mismatch: {actual} != {expected_sha256}"
        )
    return actual


def verify_kaggle_adapter(handle: str, expected_sha256: str) -> str:
    try:
        import kagglehub
    except ImportError as error:
        raise RuntimeError(
            "Install release dependencies with `pip install -e .[release]`."
        ) from error
    with tempfile.TemporaryDirectory(prefix="falsifyrl-kaggle-model-") as temporary:
        downloaded = Path(
            kagglehub.model_download(
                handle,
                path="adapter_model.safetensors",
                force_download=True,
                output_dir=temporary,
            )
        )
        if downloaded.is_dir():
            downloaded = downloaded / "adapter_model.safetensors"
        actual = _sha256(downloaded)
    if actual != expected_sha256:
        raise ValueError(f"Kaggle adapter hash mismatch: {actual} != {expected_sha256}")
    return actual


def update_private_manifest(
    manifest_path: Path,
    *,
    huggingface_model_url: str,
    kaggle_model_url: str,
    space_url: str,
    evaluation_report_url: str,
) -> None:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["links"]["huggingface_model"] = huggingface_model_url
    manifest["links"]["kaggle_model"] = kaggle_model_url
    manifest["links"]["huggingface_space"] = space_url
    manifest["links"]["evaluation_report"] = evaluation_report_url
    manifest["attestations"]["weights_public_on_both_platforms"] = True
    temporary = manifest_path.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(manifest_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Wait for a passing held-out comparison, publish verified weights "
            "to both model hosts, and publish the Hugging Face Space."
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
        "--comparison",
        type=Path,
        default=Path("outputs/evaluation/comparison.json"),
    )
    parser.add_argument(
        "--submission-manifest",
        type=Path,
        default=Path("outputs/submission/manifest.json"),
    )
    parser.add_argument(
        "--model-bundle",
        type=Path,
        default=Path("artifacts/release/model"),
    )
    parser.add_argument(
        "--space-bundle",
        type=Path,
        default=Path("artifacts/release/space"),
    )
    parser.add_argument(
        "--test-jsonl",
        type=Path,
        default=Path("outputs/falsifyrl_seed_v1/test.jsonl"),
    )
    parser.add_argument("--huggingface-owner")
    parser.add_argument("--kaggle-owner")
    parser.add_argument("--model-slug", default="falsifyrl-autoscientist")
    parser.add_argument("--space-slug", default="falsifyrl")
    parser.add_argument("--poll-seconds", type=float, default=30.0)
    parser.add_argument("--timeout-seconds", type=float, default=172_800.0)
    return parser.parse_args()


def main() -> None:
    load_dotenv()
    args = parse_args()
    huggingface_owner = args.huggingface_owner or os.environ.get("FALSIFYRL_HF_OWNER")
    kaggle_owner = args.kaggle_owner or os.environ.get("FALSIFYRL_KAGGLE_OWNER")
    if not huggingface_owner or not kaggle_owner:
        raise RuntimeError("both Hugging Face and Kaggle owners are required")

    state, _comparison, submission = await_passing_evaluation(
        args.state,
        args.comparison,
        args.submission_manifest,
        poll_seconds=args.poll_seconds,
        timeout_seconds=args.timeout_seconds,
    )
    dataset_repo_id = f"{huggingface_owner}/falsifyrl-adapted"
    adapter_config_candidates = list(
        Path("outputs/autoscientist/extracted-checkpoint").rglob(
            "adapter_config.json"
        )
    )
    if len(adapter_config_candidates) != 1:
        raise ValueError("evaluated checkpoint must contain one adapter_config.json")
    adapter_config = json.loads(
        adapter_config_candidates[0].read_text(encoding="utf-8")
    )
    base_model_id = str(adapter_config["base_model_name_or_path"])

    model_manifest = prepare_model_bundle(
        args.checkpoint,
        args.model_bundle,
        base_model_id=base_model_id,
        dataset_repo_id=dataset_repo_id,
        autoscientist_run_id=str(state.autoscientist_run_id),
        best_win_rate=float(state.best_win_rate),
        evaluation_report=args.comparison,
    )
    expected_sha256 = model_manifest["files"]["adapter_model.safetensors"]["sha256"]
    huggingface_model_url = publish_huggingface_model(
        args.model_bundle,
        owner=huggingface_owner,
        slug=args.model_slug,
    )
    kaggle_model_url = publish_kaggle_model(
        args.model_bundle,
        owner=kaggle_owner,
        slug=args.model_slug,
    )
    huggingface_model_id = f"{huggingface_owner}/{args.model_slug}"
    kaggle_model_handle = f"{kaggle_owner}/{args.model_slug}/pytorch/lora"
    verification = {
        "expected_sha256": expected_sha256,
        "huggingface_sha256": verify_huggingface_adapter(
            huggingface_model_id,
            expected_sha256,
        ),
        "kaggle_sha256": verify_kaggle_adapter(
            kaggle_model_handle,
            expected_sha256,
        ),
    }
    (args.model_bundle / "publication-verification.json").write_text(
        json.dumps(verification, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    prepare_space_bundle("space", args.test_jsonl, args.space_bundle)
    space_url = publish_huggingface_space(
        args.space_bundle,
        owner=huggingface_owner,
        base_model_id=base_model_id,
        model_repo_id=huggingface_model_id,
        slug=args.space_slug,
    )
    evaluation_report_url = (
        f"{huggingface_model_url}/resolve/main/evaluation-report.json"
    )
    update_private_manifest(
        args.submission_manifest,
        huggingface_model_url=huggingface_model_url,
        kaggle_model_url=kaggle_model_url,
        space_url=space_url,
        evaluation_report_url=evaluation_report_url,
    )
    print(
        json.dumps(
            {
                "huggingface_model_url": huggingface_model_url,
                "kaggle_model_url": kaggle_model_url,
                "space_url": space_url,
                "verification": verification,
                "dataset_url": submission["links"]["huggingface_dataset"],
            },
            indent=2,
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
