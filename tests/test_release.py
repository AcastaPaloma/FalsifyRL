from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from falsifyrl.autoscientist import AutoScientistPlan, WorkflowState
from falsifyrl.release import (
    DATASET_FILES,
    audit_model_bundle,
    prepare_adapted_dataset_bundle,
    prepare_dataset_bundle,
    publish_huggingface_space,
    render_model_card,
    require_huggingface_token,
    require_kaggle_token,
    set_kaggle_dataset_public,
    set_kaggle_model_public,
    set_kaggle_model_visibility,
    verify_anonymous_public_page,
)
from scripts.continue_dataset_release import await_audited_export
from scripts.continue_dataset_release import (
    update_private_manifest as update_dataset_manifest,
)
from scripts.continue_model_release import (
    await_passing_evaluation,
    resolve_evaluated_adapter_base_model,
    rollback_staged_publication,
    validate_model_release_configuration,
    verify_selected_release_artifacts,
)
from scripts.continue_model_release import (
    update_private_manifest as update_model_manifest,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_selected_release_artifacts_are_hash_bound_to_run(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "checkpoint.tgz"
    checkpoint.write_bytes(b"checkpoint")
    adapter_dir = tmp_path / "adapter"
    adapter_dir.mkdir()
    adapter = adapter_dir / "adapter_model.safetensors"
    adapter.write_bytes(b"adapter")
    base = tmp_path / "base.jsonl"
    adapted = tmp_path / "adapted.jsonl"
    base.write_text("base\n", encoding="utf-8")
    adapted.write_text("adapted\n", encoding="utf-8")
    state = WorkflowState(
        plan=AutoScientistPlan(source="file", local_file="train.jsonl"),
        autoscientist_run_id="run-123",
        resolved_model="base/model",
    )
    manifest = tmp_path / "checkpoint-manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "autoscientist_run_id": "run-123",
                "base_model_id": "base/model",
                "original_checkpoint": {"sha256": _sha256(checkpoint)},
                "adapter_model": {"sha256": _sha256(adapter)},
            }
        ),
        encoding="utf-8",
    )
    comparison = {
        "evidence": {
            "base_predictions_sha256": _sha256(base),
            "adapted_predictions_sha256": _sha256(adapted),
        }
    }

    verify_selected_release_artifacts(
        state=state,
        comparison=comparison,
        checkpoint=checkpoint,
        checkpoint_manifest=manifest,
        adapter_dir=adapter_dir,
        base_predictions=base,
        adapted_predictions=adapted,
    )

    adapted.write_text("tampered\n", encoding="utf-8")
    with pytest.raises(ValueError, match="adapted_predictions_sha256"):
        verify_selected_release_artifacts(
            state=state,
            comparison=comparison,
            checkpoint=checkpoint,
            checkpoint_manifest=manifest,
            adapter_dir=adapter_dir,
            base_predictions=base,
            adapted_predictions=adapted,
        )


def test_llama_release_configuration_fails_closed_on_apache_defaults(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="slug must begin"):
        validate_model_release_configuration(
            base_model_id="meta-llama/Llama-3.2-3B-Instruct",
            model_slug="falsifyrl-autoscientist",
            model_card_template=Path("release/model/README.md"),
            model_license_file=Path("release/model/LICENSE"),
            kaggle_license_name="Apache 2.0",
        )

    license_file = tmp_path / "LLAMA-3.2-LICENSE.txt"
    license_file.write_text(
        "LLAMA 3.2 COMMUNITY LICENSE AGREEMENT\n"
        "1. License Rights and Redistribution.\n"
        "Meta Platforms, Inc.\n",
        encoding="utf-8",
    )
    validate_model_release_configuration(
        base_model_id="meta-llama/Llama-3.2-3B-Instruct",
        model_slug="Llama-FalsifyRL-AutoScientist",
        model_card_template=Path("release/model/README.llama3.2.md"),
        model_license_file=license_file,
        kaggle_license_name=None,
    )


def test_llama_release_rejects_an_unrelated_license(tmp_path: Path) -> None:
    license_file = tmp_path / "LICENSE"
    license_file.write_text("Apache License 2.0\n", encoding="utf-8")

    with pytest.raises(ValueError, match="expected Llama 3.2"):
        validate_model_release_configuration(
            base_model_id="meta-llama/Llama-3.2-3B-Instruct",
            model_slug="Llama-FalsifyRL-AutoScientist",
            model_card_template=Path("release/model/README.llama3.2.md"),
            model_license_file=license_file,
            kaggle_license_name=None,
        )


def _fake_dataset(source: Path) -> None:
    source.mkdir()
    file_manifest = {}
    for filename in DATASET_FILES:
        if filename == "manifest.json":
            continue
        path = source / filename
        content = (
            "prompt,completion\np,c\n"
            if filename == "train.csv"
            else f"{filename}\n"
        )
        path.write_text(content, encoding="utf-8")
        file_manifest[filename] = {
            "bytes": path.stat().st_size,
            "sha256": _sha256(path),
        }
    manifest = {
        "dataset_version": "1.0.0",
        "files": file_manifest,
        "validation": {
            "case_count": 8,
            "split_counts": {"train": 4, "validation": 2, "test": 2},
        },
    }
    (source / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")


def test_release_bundle_rechecks_source_hashes(tmp_path: Path) -> None:
    source = tmp_path / "source"
    bundle = tmp_path / "bundle"
    card = tmp_path / "card.md"
    license_file = tmp_path / "LICENSE"
    _fake_dataset(source)
    card.write_text("# Card\n", encoding="utf-8")
    license_file.write_text("MIT\n", encoding="utf-8")

    release_manifest = prepare_dataset_bundle(
        source,
        bundle,
        card_path=card,
        license_path=license_file,
    )

    assert release_manifest["case_count"] == 8
    assert (bundle / "README.md").read_text(encoding="utf-8") == "# Card\n"
    assert set(release_manifest["files"]) == set(DATASET_FILES) | {"README.md", "LICENSE"}


def test_release_bundle_rejects_tampering(tmp_path: Path) -> None:
    source = tmp_path / "source"
    _fake_dataset(source)
    (source / "train.csv").write_text("tampered", encoding="utf-8")

    with pytest.raises(ValueError, match="source hash mismatch"):
        prepare_dataset_bundle(source, tmp_path / "bundle")


def test_adapted_release_requires_exact_audited_training_file(tmp_path: Path) -> None:
    source = tmp_path / "source"
    bundle = tmp_path / "bundle"
    card = tmp_path / "card.md"
    license_file = tmp_path / "LICENSE"
    adapted = tmp_path / "adapted.csv"
    audit_path = tmp_path / "adapted.audit.json"
    _fake_dataset(source)
    manifest_path = source / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["validation"]["split_counts"]["train"] = 1
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    adapted.write_text("enhanced_prompt,enhanced_completion\np,c\n", encoding="utf-8")
    audit = {
        "dataset_variant": "adapted",
        "row_count": 1,
        "source_row_count": 1,
        "source_unique_row_count": 1,
        "exact_duplicate_rows_collapsed": 0,
        "all_source_prompts_matched": True,
        "all_completions_strict_json": True,
        "all_diagnosis_invariants_preserved": True,
        "source_sha256": _sha256(source / "train.csv"),
        "adapted_sha256": _sha256(adapted),
        "dataset_id": "dataset-123",
        "adaptation_run_id": "run-456",
    }
    audit_path.write_text(json.dumps(audit), encoding="utf-8")
    card.write_text("# Adapted\n", encoding="utf-8")
    license_file.write_text("MIT\n", encoding="utf-8")

    release_manifest = prepare_adapted_dataset_bundle(
        source,
        adapted,
        audit_path,
        bundle,
        card_path=card,
        license_path=license_file,
    )

    assert release_manifest["dataset_variant"] == "adapted"
    assert release_manifest["training_file_sha256"] == _sha256(adapted)
    assert release_manifest["training_row_count"] == 1
    assert release_manifest["source_training_row_count"] == 1
    assert (bundle / "train.csv").read_bytes() == adapted.read_bytes()
    assert (bundle / "source_train.csv").read_bytes() == (source / "train.csv").read_bytes()

    adapted.write_text("tampered", encoding="utf-8")
    with pytest.raises(ValueError, match="adapted CSV hash"):
        prepare_adapted_dataset_bundle(
            source,
            adapted,
            audit_path,
            tmp_path / "second-bundle",
            card_path=card,
            license_path=license_file,
        )


def test_release_tokens_are_environment_only(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "KAGGLE_API_TOKEN"):
        monkeypatch.delenv(name, raising=False)
    with pytest.raises(RuntimeError, match="HF_TOKEN"):
        require_huggingface_token()
    with pytest.raises(RuntimeError, match="KAGGLE_API_TOKEN"):
        require_kaggle_token()


def test_model_card_requires_all_final_run_values(tmp_path: Path) -> None:
    template = tmp_path / "template.md"
    destination = tmp_path / "README.md"
    template.write_text(
        "BASE_MODEL_ID DATASET_REPO_ID AUTOSCIENTIST_RUN_ID "
        "BEST_WIN_RATE EVALUATION_REPORT_URL",
        encoding="utf-8",
    )

    render_model_card(
        template,
        destination,
        base_model_id="base/model",
        dataset_repo_id="owner/data",
        autoscientist_run_id="run-1",
        best_win_rate=0.81234,
        evaluation_report_url="https://example.test/report",
    )

    content = destination.read_text(encoding="utf-8")
    assert "BASE_MODEL_ID" not in content
    assert "0.8123" in content


def test_model_bundle_audit_rejects_placeholders_and_missing_weights(
    tmp_path: Path,
) -> None:
    (tmp_path / "README.md").write_text("BASE_MODEL_ID", encoding="utf-8")
    (tmp_path / "adapter_config.json").write_text("{}", encoding="utf-8")
    (tmp_path / "LICENSE").write_text("Apache-2.0", encoding="utf-8")

    with pytest.raises(ValueError, match="missing required files"):
        audit_model_bundle(tmp_path)

    (tmp_path / "adapter_model.safetensors").write_bytes(b"weights")
    with pytest.raises(ValueError, match="unresolved markers"):
        audit_model_bundle(tmp_path)


def test_publishable_space_requires_cached_predictions(tmp_path: Path) -> None:
    for filename in ("README.md", "app.py", "requirements.txt"):
        (tmp_path / filename).write_text(filename, encoding="utf-8")
    (tmp_path / "examples.json").write_text(
        json.dumps([{"example_id": f"example-{index}"} for index in range(16)]),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="predictions.json"):
        publish_huggingface_space(
            tmp_path,
            owner="owner",
            base_model_id="base/model",
            model_repo_id="owner/model",
        )


def test_kaggle_publication_explicitly_sets_dataset_and_model_visibility(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured = {}

    class DatasetApi:
        def update_dataset_metadata(self, request):
            captured["dataset"] = request
            return SimpleNamespace(errors=[])

    class ModelApi:
        def update_model(self, request):
            captured["model"] = request
            return SimpleNamespace(error="")

    client = SimpleNamespace(
        datasets=SimpleNamespace(dataset_api_client=DatasetApi()),
        models=SimpleNamespace(model_api_client=ModelApi()),
    )

    class ClientContext:
        def __enter__(self):
            return client

        def __exit__(self, *args):
            return False

    import kagglehub.clients

    monkeypatch.setattr(
        kagglehub.clients,
        "build_kaggle_client",
        lambda: ClientContext(),
    )

    set_kaggle_dataset_public(
        owner="owner",
        slug="data",
        title="Data",
        subtitle="Subtitle",
        description="Description",
    )
    set_kaggle_model_public(
        owner="owner",
        slug="model",
        title="Model",
        subtitle="Subtitle",
        description="Description",
    )

    assert captured["dataset"].settings.is_private is False
    assert captured["dataset"].settings.licenses[0].name == "MIT"
    assert captured["model"].is_private is False
    assert "is_private" in captured["model"].update_mask.paths


def test_kaggle_model_visibility_can_restore_private_after_failed_release(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured = {}

    class ModelApi:
        def update_model(self, request):
            captured["model"] = request
            return SimpleNamespace(error="")

    client = SimpleNamespace(models=SimpleNamespace(model_api_client=ModelApi()))

    class ClientContext:
        def __enter__(self):
            return client

        def __exit__(self, *args):
            return False

    import kagglehub.clients

    monkeypatch.setattr(
        kagglehub.clients,
        "build_kaggle_client",
        lambda: ClientContext(),
    )
    set_kaggle_model_visibility(
        owner="owner",
        slug="model",
        title="Model",
        subtitle="Subtitle",
        description="Description",
        private=True,
    )

    assert captured["model"].is_private is True
    assert "is_private" in captured["model"].update_mask.paths


def test_failed_release_rolls_back_only_created_artifacts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = []

    monkeypatch.setattr(
        "scripts.continue_model_release.set_huggingface_repo_visibility",
        lambda repo_id, **kwargs: calls.append(("hf", repo_id, kwargs)),
    )
    monkeypatch.setattr(
        "scripts.continue_model_release.set_kaggle_model_visibility",
        lambda **kwargs: calls.append(("kaggle", kwargs)),
    )

    failures = rollback_staged_publication(
        huggingface_model_id="owner/model",
        huggingface_space_id="owner/space",
        kaggle_owner="owner",
        model_slug="model",
        model_created=True,
        space_created=True,
        kaggle_model_created=True,
    )

    assert failures == []
    assert calls[0] == ("hf", "owner/space", {"repo_type": "space", "private": True})
    assert calls[1] == ("hf", "owner/model", {"repo_type": "model", "private": True})
    assert calls[2][0] == "kaggle"
    assert calls[2][1]["private"] is True


def test_public_page_verification_rejects_private_or_wrong_artifact() -> None:
    public = SimpleNamespace(
        status_code=200,
        url="https://www.kaggle.com/datasets/owner/falsifyrl-adapted",
        text="<title>FalsifyRL Adapted</title>",
    )
    verify_anonymous_public_page(
        public.url,
        expected_marker="falsifyrl",
        fetcher=lambda *args, **kwargs: public,
    )

    private = SimpleNamespace(
        status_code=404,
        url="https://www.kaggle.com/datasets/owner/private",
        text="",
    )
    with pytest.raises(RuntimeError, match="HTTP 404"):
        verify_anonymous_public_page(
            private.url,
            expected_marker="falsifyrl",
            fetcher=lambda *args, **kwargs: private,
        )

    wrong = SimpleNamespace(
        status_code=200,
        url="https://www.kaggle.com/datasets/owner/other",
        text="<title>Another dataset</title>",
    )
    with pytest.raises(RuntimeError, match="expected marker"):
        verify_anonymous_public_page(
            wrong.url,
            expected_marker="falsifyrl",
            fetcher=lambda *args, **kwargs: wrong,
        )


def test_dataset_release_waits_for_exact_audited_export(tmp_path: Path) -> None:
    adapted = tmp_path / "adapted.csv"
    audit = tmp_path / "adapted.audit.json"
    adapted.write_text("prompt,completion\n", encoding="utf-8")
    audit.write_text("{}\n", encoding="utf-8")
    state_path = tmp_path / "workflow.json"
    WorkflowState(
        plan=AutoScientistPlan(
            source="file",
            local_file="train.jsonl",
            expected_training_rows=2,
        ),
        dataset_id="dataset-123",
        dataset_status="succeeded",
        adaptation_run_id="run-456",
        adapted_export_path=str(adapted),
        adapted_export_sha256=_sha256(adapted),
        adapted_audit_path=str(audit),
        adapted_row_count=1,
        adapted_schema_valid=True,
    ).save(state_path)

    result = await_audited_export(
        state_path,
        poll_seconds=0,
        timeout_seconds=1,
    )

    assert result.adapted_export_sha256 == _sha256(adapted)


def test_dataset_release_updates_only_proven_submission_fields(tmp_path: Path) -> None:
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "links": {"huggingface_dataset": None, "kaggle_dataset": None},
                "dataset": {"variant": None, "sha256_manifest": None},
                "attestations": {
                    "dataset_public_on_both_platforms": None,
                    "same_dataset_used_for_training": None,
                    "at_least_18": None,
                },
            }
        ),
        encoding="utf-8",
    )

    update_dataset_manifest(
        manifest_path,
        huggingface_url="https://huggingface.co/datasets/owner/falsifyrl-adapted",
        kaggle_url="https://www.kaggle.com/datasets/owner/falsifyrl-adapted",
        manifest_url=(
            "https://huggingface.co/datasets/owner/falsifyrl-adapted/"
            "resolve/main/release-manifest.json"
        ),
    )
    updated = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert updated["dataset"]["variant"] == "adapted"
    assert updated["attestations"]["dataset_public_on_both_platforms"] is True
    assert updated["attestations"]["same_dataset_used_for_training"] is True
    assert updated["attestations"]["at_least_18"] is None


def test_model_release_requires_passing_evaluation_and_published_dataset(
    tmp_path: Path,
) -> None:
    state_path = tmp_path / "workflow.json"
    WorkflowState(
        plan=AutoScientistPlan(
            source="file",
            local_file="train.jsonl",
            expected_training_rows=1,
        ),
        autoscientist_run_id="experiment-123",
        autoscientist_status="succeeded",
        best_win_rate=0.8,
        resolved_model="meta-llama/Llama-3.2-3B-Instruct",
    ).save(state_path)
    comparison_path = tmp_path / "comparison.json"
    comparison_path.write_text(
        json.dumps(
            {
                "evidence": {
                    "autoscientist_run_id": "experiment-123",
                    "base_model_id": "meta-llama/Llama-3.2-3B-Instruct",
                },
                "metrics": {
                    "base": {"composite_score": 0.0},
                    "adapted": {
                        "composite_score": 0.8,
                        "json_validity": 0.99,
                    },
                }
            }
        ),
        encoding="utf-8",
    )
    submission_path = tmp_path / "submission.json"
    submission_path.write_text(
        json.dumps(
            {
                "links": {
                    "huggingface_dataset": "https://huggingface.co/datasets/owner/data",
                    "kaggle_dataset": "https://www.kaggle.com/datasets/owner/data",
                }
            }
        ),
        encoding="utf-8",
    )

    state, comparison, _submission = await_passing_evaluation(
        state_path,
        comparison_path,
        submission_path,
        poll_seconds=0,
        timeout_seconds=1,
    )

    assert state.autoscientist_run_id == "experiment-123"
    assert comparison["metrics"]["adapted"]["composite_score"] == 0.8


def test_model_release_rejects_comparison_for_different_run_or_base(
    tmp_path: Path,
) -> None:
    state_path = tmp_path / "workflow.json"
    WorkflowState(
        plan=AutoScientistPlan(source="file", local_file="train.jsonl"),
        autoscientist_run_id="llama-run",
        autoscientist_status="succeeded",
        best_win_rate=0.8,
        resolved_model="meta-llama/Llama-3.2-3B-Instruct",
    ).save(state_path)
    comparison_path = tmp_path / "comparison.json"
    comparison_path.write_text(
        json.dumps(
            {
                "evidence": {
                    "autoscientist_run_id": "qwen-run",
                    "base_model_id": "Qwen/Qwen3.5-9B",
                },
                "metrics": {
                    "base": {"composite_score": 0.0},
                    "adapted": {"composite_score": 0.8, "json_validity": 0.99},
                },
            }
        ),
        encoding="utf-8",
    )
    submission_path = tmp_path / "submission.json"
    submission_path.write_text(
        json.dumps(
            {
                "links": {
                    "huggingface_dataset": "https://huggingface.co/datasets/owner/data",
                    "kaggle_dataset": "https://www.kaggle.com/datasets/owner/data",
                }
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="run ID"):
        await_passing_evaluation(
            state_path,
            comparison_path,
            submission_path,
            poll_seconds=0,
            timeout_seconds=1,
        )


def test_model_release_canonicalizes_only_matching_adapter_base_model(
    tmp_path: Path,
) -> None:
    adapter_dir = tmp_path / "adapter"
    adapter_dir.mkdir()
    config_path = adapter_dir / "adapter_config.json"
    config_path.write_text(
        json.dumps(
            {
                "base_model_name_or_path": (
                    "meta-llama/Llama-3.2-3B-Instruct-reference__tog__ft"
                )
            }
        ),
        encoding="utf-8",
    )
    state = WorkflowState(
        plan=AutoScientistPlan(source="file", local_file="train.jsonl"),
        resolved_model="meta-llama/Llama-3.2-3B-Instruct",
    )

    assert resolve_evaluated_adapter_base_model(adapter_dir, state) == state.resolved_model
    assert json.loads(config_path.read_text(encoding="utf-8"))["base_model_name_or_path"] == (
        state.resolved_model
    )

    config_path.write_text(
        json.dumps({"base_model_name_or_path": "Qwen/Qwen3.5-9B"}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="does not match"):
        resolve_evaluated_adapter_base_model(adapter_dir, state)


def test_model_release_records_links_without_claiming_runtime_success(
    tmp_path: Path,
) -> None:
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "links": {
                    "huggingface_model": None,
                    "kaggle_model": None,
                    "huggingface_space": None,
                    "evaluation_report": None,
                },
                "attestations": {
                    "weights_public_on_both_platforms": None,
                    "at_least_18": None,
                },
            }
        ),
        encoding="utf-8",
    )

    update_model_manifest(
        manifest_path,
        huggingface_model_url="https://huggingface.co/owner/model",
        kaggle_model_url="https://www.kaggle.com/models/owner/model/pytorch/lora",
        space_url="https://huggingface.co/spaces/owner/falsifyrl",
        evaluation_report_url=(
            "https://huggingface.co/owner/model/resolve/main/evaluation-report.json"
        ),
    )
    updated = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert updated["attestations"]["weights_public_on_both_platforms"] is True
    assert updated["attestations"]["at_least_18"] is None
