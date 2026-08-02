from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from dotenv import load_dotenv

from falsifyrl.release import (
    audit_model_bundle,
    publish_huggingface_space,
    set_huggingface_repo_visibility,
    verify_anonymous_public_page,
)
from scripts.continue_model_release import (
    await_passing_evaluation,
    extract_adapter_checkpoint,
    resolve_evaluated_adapter_base_model,
    update_private_manifest,
    verify_huggingface_adapter,
    verify_kaggle_adapter,
    verify_selected_release_artifacts,
    write_selected_release_record,
)


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def audit_existing_model_bundle(
    bundle_dir: Path,
    *,
    state_run_id: str,
    base_model_id: str,
    comparison: dict,
) -> str:
    """Verify the already-uploaded bundle before resuming a staged release."""
    audit_model_bundle(bundle_dir)
    manifest_path = bundle_dir / "release-manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("autoscientist_run_id") != state_run_id:
        raise ValueError("model bundle run ID does not match selected run")
    if manifest.get("base_model_id") != base_model_id:
        raise ValueError("model bundle base model does not match selected run")

    evidence = comparison.get("evidence", {})
    required_files = {
        "adapter_model.safetensors": evidence.get("adapter_sha256"),
        "falsifyrl-base-test-predictions.jsonl": evidence.get(
            "base_predictions_sha256"
        ),
        "falsifyrl-adapted-test-predictions.jsonl": evidence.get(
            "adapted_predictions_sha256"
        ),
        "evaluation-manifest.json": None,
        "colab-evaluation.json": None,
    }
    manifest_files = manifest.get("files")
    if not isinstance(manifest_files, dict):
        raise ValueError("model bundle release manifest has no file inventory")
    for filename, expected_evidence_sha in required_files.items():
        path = bundle_dir / filename
        if not path.is_file() or filename not in manifest_files:
            raise ValueError(f"model bundle is missing immutable evidence: {filename}")
        actual = _sha256(path)
        if manifest_files[filename].get("sha256") != actual:
            raise ValueError(f"model bundle manifest hash mismatch for {filename}")
        if expected_evidence_sha is not None and expected_evidence_sha != actual:
            raise ValueError(f"model bundle comparison hash mismatch for {filename}")

    evaluation_manifest = json.loads(
        (bundle_dir / "evaluation-manifest.json").read_text(encoding="utf-8")
    )
    evaluation_report = json.loads(
        (bundle_dir / "colab-evaluation.json").read_text(encoding="utf-8")
    )
    evaluation_bindings = {
        "autoscientist_run_id": state_run_id,
        "base_model_id": base_model_id,
        "adapter_sha256": required_files["adapter_model.safetensors"],
        "example_count": 640,
    }
    for key, expected in evaluation_bindings.items():
        if evaluation_manifest.get(key) != expected:
            raise ValueError(f"evaluation manifest binding mismatch for {key}")
    report_bindings = {
        "run_id": state_run_id,
        "base_model_id": base_model_id,
        "adapter_sha256": required_files["adapter_model.safetensors"],
        "example_count": 640,
        "base_predictions_sha256": required_files[
            "falsifyrl-base-test-predictions.jsonl"
        ],
        "adapted_predictions_sha256": required_files[
            "falsifyrl-adapted-test-predictions.jsonl"
        ],
    }
    for key, expected in report_bindings.items():
        if evaluation_report.get(key) != expected:
            raise ValueError(f"Colab evaluation binding mismatch for {key}")
    return str(required_files["adapter_model.safetensors"])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Resume an exact staged model release after the existing Kaggle model "
            "has been made public manually. Never uploads another model version."
        )
    )
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--adapter-dir", type=Path, required=True)
    parser.add_argument("--checkpoint-manifest", type=Path, required=True)
    parser.add_argument("--comparison", type=Path, required=True)
    parser.add_argument("--submission-manifest", type=Path, required=True)
    parser.add_argument("--model-bundle", type=Path, required=True)
    parser.add_argument("--space-bundle", type=Path, required=True)
    parser.add_argument("--base-predictions", type=Path, required=True)
    parser.add_argument("--model-predictions", type=Path, required=True)
    parser.add_argument("--huggingface-owner", required=True)
    parser.add_argument("--kaggle-owner", required=True)
    parser.add_argument("--model-slug", required=True)
    parser.add_argument("--kaggle-model-version", type=int, required=True)
    parser.add_argument("--selected-release-record", type=Path, required=True)
    parser.add_argument("--space-slug", required=True)
    return parser.parse_args()


def main() -> None:
    load_dotenv()
    args = parse_args()
    if args.selected_release_record.exists():
        raise ValueError(
            "selected release record already exists; refusing to resume publication"
        )
    state, comparison, submission = await_passing_evaluation(
        args.state,
        args.comparison,
        args.submission_manifest,
        poll_seconds=0,
        timeout_seconds=1,
    )
    extract_adapter_checkpoint(args.checkpoint, args.adapter_dir)
    base_model_id = resolve_evaluated_adapter_base_model(args.adapter_dir, state)
    verify_selected_release_artifacts(
        state=state,
        comparison=comparison,
        checkpoint=args.checkpoint,
        checkpoint_manifest=args.checkpoint_manifest,
        adapter_dir=args.adapter_dir,
        base_predictions=args.base_predictions,
        adapted_predictions=args.model_predictions,
    )
    expected_sha256 = audit_existing_model_bundle(
        args.model_bundle,
        state_run_id=str(state.autoscientist_run_id),
        base_model_id=base_model_id,
        comparison=comparison,
    )

    huggingface_model_id = f"{args.huggingface_owner}/{args.model_slug}"
    kaggle_model_handle = (
        f"{args.kaggle_owner}/{args.model_slug}/pytorch/lora/"
        f"{args.kaggle_model_version}"
    )
    huggingface_space_id = f"{args.huggingface_owner}/{args.space_slug}"
    huggingface_model_url = f"https://huggingface.co/{huggingface_model_id}"
    kaggle_model_url = (
        f"https://www.kaggle.com/models/{args.kaggle_owner}/"
        f"{args.model_slug}/pytorch/lora"
    )
    space_url = f"https://huggingface.co/spaces/{huggingface_space_id}"

    verification = {
        "expected_sha256": expected_sha256,
        "huggingface_sha256": verify_huggingface_adapter(
            huggingface_model_id,
            expected_sha256,
            private=True,
        ),
        "kaggle_sha256": verify_kaggle_adapter(
            kaggle_model_handle,
            expected_sha256,
        ),
    }
    verify_anonymous_public_page(
        kaggle_model_url,
        expected_marker=args.model_slug,
    )
    (args.model_bundle / "publication-verification.json").write_text(
        json.dumps(verification, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    space_created = False
    model_promoted = False
    try:
        publish_huggingface_space(
            args.space_bundle,
            owner=args.huggingface_owner,
            base_model_id=base_model_id,
            model_repo_id=huggingface_model_id,
            slug=args.space_slug,
            private=True,
        )
        space_created = True
        set_huggingface_repo_visibility(
            huggingface_model_id,
            repo_type="model",
            private=False,
        )
        model_promoted = True
        verify_anonymous_public_page(
            huggingface_model_url,
            expected_marker=args.model_slug,
        )
        set_huggingface_repo_visibility(
            huggingface_space_id,
            repo_type="space",
            private=False,
        )
        verify_anonymous_public_page(space_url, expected_marker=args.space_slug)
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
        selected_release = write_selected_release_record(
            args.selected_release_record,
            state=state,
            comparison=comparison,
            adapter_sha256=expected_sha256,
            huggingface_owner=args.huggingface_owner,
            kaggle_owner=args.kaggle_owner,
            model_slug=args.model_slug,
            kaggle_model_version=args.kaggle_model_version,
            huggingface_model_url=huggingface_model_url,
            kaggle_model_url=kaggle_model_url,
        )
    except Exception as error:
        rollback_failures = []
        if space_created:
            try:
                set_huggingface_repo_visibility(
                    huggingface_space_id,
                    repo_type="space",
                    private=True,
                )
            except Exception as rollback_error:
                rollback_failures.append(str(rollback_error))
        if model_promoted:
            try:
                set_huggingface_repo_visibility(
                    huggingface_model_id,
                    repo_type="model",
                    private=True,
                )
            except Exception as rollback_error:
                rollback_failures.append(str(rollback_error))
        if rollback_failures:
            raise RuntimeError(
                "resume failed and Hugging Face rollback was incomplete: "
                + "; ".join(rollback_failures)
            ) from error
        raise

    print(
        json.dumps(
            {
                "huggingface_model_url": huggingface_model_url,
                "kaggle_model_url": kaggle_model_url,
                "space_url": space_url,
                "verification": verification,
                "selected_release_record": str(
                    args.selected_release_record.resolve()
                ),
                "kaggle_model_source": selected_release["kaggle_model"]["source"],
                "dataset_url": submission["links"]["huggingface_dataset"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
