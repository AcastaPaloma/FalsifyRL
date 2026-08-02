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
    canonicalize_adapter_base_model,
    extract_adapter_checkpoint,
    prepare_model_bundle,
    publish_huggingface_model,
    publish_huggingface_space,
    publish_kaggle_model,
    set_huggingface_repo_visibility,
    set_kaggle_model_visibility,
    verify_anonymous_public_page,
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
            and state.resolved_model
            and state.best_win_rate is not None
            and comparison_path.is_file()
            and submission["links"].get("huggingface_dataset")
            and submission["links"].get("kaggle_dataset")
        ):
            comparison = json.loads(comparison_path.read_text(encoding="utf-8"))
            evidence = comparison.get("evidence")
            if not isinstance(evidence, dict):
                raise RuntimeError("comparison is missing identity evidence")
            if evidence.get("autoscientist_run_id") != state.autoscientist_run_id:
                raise RuntimeError("comparison run ID does not match workflow state")
            if evidence.get("base_model_id") != state.resolved_model:
                raise RuntimeError("comparison base model does not match workflow state")
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


def resolve_evaluated_adapter_base_model(
    adapter_dir: Path,
    state: WorkflowState,
) -> str:
    if not state.resolved_model:
        raise ValueError("workflow state is missing the resolved base model")
    adapter_config_candidates = list(adapter_dir.rglob("adapter_config.json"))
    if len(adapter_config_candidates) != 1:
        raise ValueError("evaluated checkpoint must contain one adapter_config.json")
    canonicalize_adapter_base_model(
        adapter_config_candidates[0].parent,
        state.resolved_model,
    )
    return state.resolved_model


def verify_selected_release_artifacts(
    *,
    state: WorkflowState,
    comparison: dict,
    checkpoint: Path,
    checkpoint_manifest: Path,
    adapter_dir: Path,
    base_predictions: Path,
    adapted_predictions: Path,
) -> None:
    manifest = json.loads(checkpoint_manifest.read_text(encoding="utf-8"))
    bindings = {
        "autoscientist_run_id": state.autoscientist_run_id,
        "base_model_id": state.resolved_model,
    }
    for key, expected in bindings.items():
        if not expected or manifest.get(key) != expected:
            raise ValueError(f"checkpoint manifest binding mismatch for {key}")
    original_checkpoint = manifest.get("original_checkpoint", {})
    if original_checkpoint.get("sha256") != _sha256(checkpoint):
        raise ValueError("selected checkpoint archive hash does not match run manifest")
    adapter_model = manifest.get("adapter_model", {})
    if adapter_model.get("sha256") != _sha256(
        adapter_dir / "adapter_model.safetensors"
    ):
        raise ValueError("selected adapter hash does not match run manifest")

    evidence = comparison.get("evidence", {})
    prediction_bindings = {
        "base_predictions_sha256": _sha256(base_predictions),
        "adapted_predictions_sha256": _sha256(adapted_predictions),
    }
    for key, actual in prediction_bindings.items():
        if evidence.get(key) != actual:
            raise ValueError(f"released prediction hash mismatch for {key}")


def verify_huggingface_adapter(
    repo_id: str,
    expected_sha256: str,
    *,
    private: bool = False,
) -> str:
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as error:
        raise RuntimeError(
            "Install release dependencies with `pip install -e .[release]`."
        ) from error
    downloaded = hf_hub_download(
        repo_id=repo_id,
        filename="adapter_model.safetensors",
        token=(
            (os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN"))
            if private
            else False
        ),
        force_download=True,
    )
    actual = _sha256(downloaded)
    if actual != expected_sha256:
        raise ValueError(
            f"Hugging Face adapter hash mismatch: {actual} != {expected_sha256}"
        )
    return actual


def rollback_staged_publication(
    *,
    huggingface_model_id: str,
    huggingface_space_id: str,
    kaggle_owner: str,
    model_slug: str,
    model_created: bool,
    space_created: bool,
    kaggle_model_created: bool,
) -> list[str]:
    """Best-effort rollback; keep failed staged artifacts private rather than public."""
    failures: list[str] = []
    actions = []
    if space_created:
        actions.append(
            lambda: set_huggingface_repo_visibility(
                huggingface_space_id, repo_type="space", private=True
            )
        )
    if model_created:
        actions.append(
            lambda: set_huggingface_repo_visibility(
                huggingface_model_id, repo_type="model", private=True
            )
        )
    if kaggle_model_created:
        actions.append(
            lambda: set_kaggle_model_visibility(
                owner=kaggle_owner,
                slug=model_slug,
                title=model_slug.replace("-", " "),
                subtitle=(
                    "LoRA critic for evidence-grounded diagnosis and executable reward repair"
                ),
                description=(
                    "Best audited AutoScientist checkpoint trained on the exact FalsifyRL "
                    "adapted dataset and evaluated on a family-disjoint held-out robotics split."
                ),
                private=True,
            )
        )
    for action in actions:
        try:
            action()
        except Exception as error:  # rollback must not hide the primary failure
            failures.append(f"{type(error).__name__}: {error}")
    return failures


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


def write_selected_release_record(
    path: Path,
    *,
    state: WorkflowState,
    comparison: dict,
    adapter_sha256: str,
    huggingface_owner: str,
    kaggle_owner: str,
    model_slug: str,
    kaggle_model_version: int,
    huggingface_model_url: str,
    kaggle_model_url: str,
) -> dict:
    """Write the one-time record that authorizes the public Kaggle evaluator."""
    if path.exists():
        raise ValueError(f"selected release record must not already exist: {path}")
    evidence = comparison.get("evidence", {})
    required_evidence = (
        "base_predictions_sha256",
        "adapted_predictions_sha256",
    )
    if not all(isinstance(evidence.get(key), str) for key in required_evidence):
        raise ValueError("comparison is missing prediction hash evidence")
    if kaggle_model_version < 1:
        raise ValueError("Kaggle model version must be positive")

    kaggle_source = (
        f"{kaggle_owner}/{model_slug}/pytorch/lora/{kaggle_model_version}"
    )
    value = {
        "schema_version": 1,
        "identity": {
            "autoscientist_run_id": state.autoscientist_run_id,
            "base_model_id": state.resolved_model,
            "adapter_sha256": adapter_sha256,
            "base_predictions_sha256": evidence["base_predictions_sha256"],
            "adapted_predictions_sha256": evidence["adapted_predictions_sha256"],
        },
        "huggingface_model": {
            "owner": huggingface_owner,
            "slug": model_slug,
            "repo_id": f"{huggingface_owner}/{model_slug}",
            "url": huggingface_model_url,
        },
        "kaggle_model": {
            "owner": kaggle_owner,
            "slug": model_slug,
            "variation": "lora",
            "version": kaggle_model_version,
            "source": kaggle_source,
            "url": kaggle_model_url,
        },
        "kaggle_dataset": {
            "owner": kaggle_owner,
            "slug": "falsifyrl-adapted",
            "url": (
                f"https://www.kaggle.com/datasets/{kaggle_owner}/falsifyrl-adapted"
            ),
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return value


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


def validate_model_release_configuration(
    *,
    base_model_id: str,
    model_slug: str,
    model_card_template: Path,
    model_license_file: Path,
    kaggle_license_name: str | None,
) -> None:
    if not base_model_id.casefold().startswith("meta-llama/"):
        return
    if not model_slug.casefold().startswith("llama"):
        raise ValueError("a Llama-derived release slug must begin with 'Llama'")
    if model_card_template == Path("release/model/README.md"):
        raise ValueError("Llama release requires the Llama-specific model card")
    if model_license_file == Path("release/model/LICENSE"):
        raise ValueError("Llama release requires the Meta community license file")
    if not model_license_file.is_file():
        raise FileNotFoundError(model_license_file)
    license_text = model_license_file.read_text(
        encoding="utf-8",
        errors="replace",
    ).casefold()
    required_license_markers = (
        "llama 3.2 community license agreement",
        "license rights and redistribution",
        "meta platforms",
    )
    if not all(marker in license_text for marker in required_license_markers):
        raise ValueError(
            "Llama release license file does not contain the expected "
            "Llama 3.2 Community License terms"
        )
    if kaggle_license_name == "Apache 2.0":
        raise ValueError("Llama release cannot use Apache 2.0 Kaggle metadata")


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
        "--adapter-dir",
        type=Path,
        required=True,
        help="empty run-scoped directory used to verify the selected checkpoint",
    )
    parser.add_argument(
        "--checkpoint-manifest",
        type=Path,
        required=True,
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
    parser.add_argument(
        "--base-predictions",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--evaluation-evidence-dir",
        type=Path,
        required=True,
        help="verified staged-evidence directory containing immutable evaluation metadata",
    )
    parser.add_argument(
        "--model-predictions",
        type=Path,
        default=Path("outputs/evaluation/falsifyrl-adapted-test-predictions.jsonl"),
    )
    parser.add_argument("--huggingface-owner")
    parser.add_argument("--kaggle-owner")
    parser.add_argument("--model-slug", default="falsifyrl-autoscientist")
    parser.add_argument("--kaggle-model-version", type=int, required=True)
    parser.add_argument(
        "--selected-release-record",
        type=Path,
        required=True,
        help="new write-once record consumed by continue_kaggle_notebook.py",
    )
    parser.add_argument("--space-slug", default="falsifyrl")
    parser.add_argument(
        "--model-card-template",
        type=Path,
        default=Path("release/model/README.md"),
    )
    parser.add_argument(
        "--model-license-file",
        type=Path,
        default=Path("release/model/LICENSE"),
    )
    parser.add_argument("--kaggle-license-name", default="Apache 2.0")
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

    state, comparison, submission = await_passing_evaluation(
        args.state,
        args.comparison,
        args.submission_manifest,
        poll_seconds=args.poll_seconds,
        timeout_seconds=args.timeout_seconds,
    )
    dataset_repo_id = f"{huggingface_owner}/falsifyrl-adapted"
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
    kaggle_license_name = args.kaggle_license_name or None
    validate_model_release_configuration(
        base_model_id=base_model_id,
        model_slug=args.model_slug,
        model_card_template=args.model_card_template,
        model_license_file=args.model_license_file,
        kaggle_license_name=kaggle_license_name,
    )
    prepare_space_bundle(
        "space",
        args.test_jsonl,
        args.space_bundle,
        prediction_jsonl=args.model_predictions,
    )

    model_manifest = prepare_model_bundle(
        args.checkpoint,
        args.model_bundle,
        base_model_id=base_model_id,
        dataset_repo_id=dataset_repo_id,
        autoscientist_run_id=str(state.autoscientist_run_id),
        best_win_rate=float(state.best_win_rate),
        evaluation_report=args.comparison,
        base_predictions=args.base_predictions,
        adapted_predictions=args.model_predictions,
        evaluation_metadata_dir=args.evaluation_evidence_dir,
        model_card_template=args.model_card_template,
        license_path=args.model_license_file,
    )
    expected_sha256 = model_manifest["files"]["adapter_model.safetensors"]["sha256"]
    huggingface_model_id = f"{huggingface_owner}/{args.model_slug}"
    kaggle_model_handle = (
        f"{kaggle_owner}/{args.model_slug}/pytorch/lora/"
        f"{args.kaggle_model_version}"
    )
    huggingface_space_id = f"{huggingface_owner}/{args.space_slug}"
    huggingface_model_url = f"https://huggingface.co/{huggingface_model_id}"
    kaggle_model_url = (
        f"https://www.kaggle.com/models/{kaggle_owner}/{args.model_slug}/pytorch/lora"
    )
    space_url = f"https://huggingface.co/spaces/{huggingface_space_id}"
    model_created = False
    kaggle_model_created = False
    space_created = False
    try:
        # Stage both weight repositories privately. Hugging Face supports an atomic
        # private create; Kaggle has no upload-visibility argument, so its helper
        # immediately requests private visibility after upload.
        publish_huggingface_model(
            args.model_bundle,
            owner=huggingface_owner,
            slug=args.model_slug,
            private=True,
        )
        model_created = True
        # The upload helper must be treated as having created a remote artifact as
        # soon as it is invoked: upload can succeed even if its immediate privacy
        # update then fails. This guarantees the outer exception path retries the
        # privacy rollback in that partial-success case.
        kaggle_model_created = True
        publish_kaggle_model(
            args.model_bundle,
            owner=kaggle_owner,
            slug=args.model_slug,
            license_name=kaggle_license_name,
            private=True,
        )
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
        (args.model_bundle / "publication-verification.json").write_text(
            json.dumps(verification, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

        # Do not expose either checkpoint until the byte-level cross-host check passes.
        set_huggingface_repo_visibility(
            huggingface_model_id, repo_type="model", private=False
        )
        set_kaggle_model_visibility(
            owner=kaggle_owner,
            slug=args.model_slug,
            title=args.model_slug.replace("-", " "),
            subtitle="LoRA critic for evidence-grounded diagnosis and executable reward repair",
            description=(
                "Best audited AutoScientist checkpoint trained on the exact FalsifyRL "
                "adapted dataset and evaluated on a family-disjoint held-out robotics split."
            ),
            private=False,
        )
        verify_anonymous_public_page(
            huggingface_model_url,
            expected_marker=args.model_slug,
        )
        verify_anonymous_public_page(
            kaggle_model_url,
            expected_marker=args.model_slug,
        )

        # The Space is also staged private, then promoted only after both model hosts
        # are public and hash-verified.
        publish_huggingface_space(
            args.space_bundle,
            owner=huggingface_owner,
            base_model_id=base_model_id,
            model_repo_id=huggingface_model_id,
            slug=args.space_slug,
            private=True,
        )
        space_created = True
        set_huggingface_repo_visibility(
            huggingface_space_id, repo_type="space", private=False
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
            huggingface_owner=huggingface_owner,
            kaggle_owner=kaggle_owner,
            model_slug=args.model_slug,
            kaggle_model_version=args.kaggle_model_version,
            huggingface_model_url=huggingface_model_url,
            kaggle_model_url=kaggle_model_url,
        )
    except Exception as error:
        rollback_failures = rollback_staged_publication(
            huggingface_model_id=huggingface_model_id,
            huggingface_space_id=huggingface_space_id,
            kaggle_owner=kaggle_owner,
            model_slug=args.model_slug,
            model_created=model_created,
            space_created=space_created,
            kaggle_model_created=kaggle_model_created,
        )
        details = "; ".join(rollback_failures)
        if details:
            raise RuntimeError(
                f"publication failed and rollback was incomplete: {details}"
            ) from error
        raise
    print(
        json.dumps(
            {
                "huggingface_model_url": huggingface_model_url,
                "kaggle_model_url": kaggle_model_url,
                "space_url": space_url,
                "verification": verification,
                "selected_release_record": str(args.selected_release_record.resolve()),
                "kaggle_model_source": selected_release["kaggle_model"]["source"],
                "dataset_url": submission["links"]["huggingface_dataset"],
            },
            indent=2,
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
