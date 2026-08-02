from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts import resume_model_release
from scripts.resume_model_release import audit_existing_model_bundle


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _existing_bundle(tmp_path: Path) -> tuple[Path, dict]:
    bundle = tmp_path / "model"
    bundle.mkdir()
    files = {
        "LICENSE": b"license",
        "README.md": b"# model\n",
        "adapter_config.json": b"{}\n",
        "adapter_model.safetensors": b"adapter",
        "falsifyrl-base-test-predictions.jsonl": b"base\n",
        "falsifyrl-adapted-test-predictions.jsonl": b"adapted\n",
    }
    for filename, content in files.items():
        (bundle / filename).write_bytes(content)
    adapter_sha = _sha256(bundle / "adapter_model.safetensors")
    base_sha = _sha256(bundle / "falsifyrl-base-test-predictions.jsonl")
    adapted_sha = _sha256(bundle / "falsifyrl-adapted-test-predictions.jsonl")
    evaluation_manifest = {
        "autoscientist_run_id": "run-1",
        "base_model_id": "base/model",
        "adapter_sha256": adapter_sha,
        "example_count": 640,
    }
    evaluation_report = {
        "run_id": "run-1",
        "base_model_id": "base/model",
        "adapter_sha256": adapter_sha,
        "example_count": 640,
        "base_predictions_sha256": base_sha,
        "adapted_predictions_sha256": adapted_sha,
    }
    (bundle / "evaluation-manifest.json").write_text(
        json.dumps(evaluation_manifest), encoding="utf-8"
    )
    (bundle / "colab-evaluation.json").write_text(
        json.dumps(evaluation_report), encoding="utf-8"
    )
    inventory = {
        path.name: {"sha256": _sha256(path)}
        for path in bundle.iterdir()
        if path.is_file()
    }
    release_manifest = {
        "autoscientist_run_id": "run-1",
        "base_model_id": "base/model",
        "files": inventory,
    }
    (bundle / "release-manifest.json").write_text(
        json.dumps(release_manifest), encoding="utf-8"
    )
    comparison = {
        "evidence": {
            "adapter_sha256": adapter_sha,
            "base_predictions_sha256": base_sha,
            "adapted_predictions_sha256": adapted_sha,
        }
    }
    return bundle, comparison


def test_existing_bundle_requires_immutable_evaluation_metadata(
    tmp_path: Path,
) -> None:
    bundle, comparison = _existing_bundle(tmp_path)

    assert audit_existing_model_bundle(
        bundle,
        state_run_id="run-1",
        base_model_id="base/model",
        comparison=comparison,
    ) == comparison["evidence"]["adapter_sha256"]

    (bundle / "evaluation-manifest.json").unlink()
    with pytest.raises(ValueError, match="immutable evidence"):
        audit_existing_model_bundle(
            bundle,
            state_run_id="run-1",
            base_model_id="base/model",
            comparison=comparison,
        )


def _resume_args(tmp_path: Path) -> SimpleNamespace:
    model_bundle = tmp_path / "model"
    model_bundle.mkdir()
    return SimpleNamespace(
        state=tmp_path / "state.json",
        checkpoint=tmp_path / "checkpoint.tgz",
        adapter_dir=tmp_path / "adapter",
        checkpoint_manifest=tmp_path / "checkpoint-manifest.json",
        comparison=tmp_path / "comparison.json",
        submission_manifest=tmp_path / "submission.json",
        model_bundle=model_bundle,
        space_bundle=tmp_path / "space",
        base_predictions=tmp_path / "base.jsonl",
        model_predictions=tmp_path / "adapted.jsonl",
        huggingface_owner="hf-owner",
        kaggle_owner="kaggle-owner",
        model_slug="Llama-FalsifyRL-AutoScientist",
        kaggle_model_version=1,
        selected_release_record=tmp_path / "selected-release.json",
        space_slug="falsifyrl-llama",
    )


def test_resume_verifies_kaggle_public_before_promoting_huggingface(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    args = _resume_args(tmp_path)
    state = SimpleNamespace(
        autoscientist_run_id="run-1",
        resolved_model="base/model",
    )
    comparison = {"evidence": {}}
    submission = {"links": {"huggingface_dataset": "https://example.test/data"}}
    calls = []
    digest = "a" * 64

    monkeypatch.setattr(resume_model_release, "load_dotenv", lambda: True)
    monkeypatch.setattr(resume_model_release, "parse_args", lambda: args)
    monkeypatch.setattr(
        resume_model_release,
        "await_passing_evaluation",
        lambda *a, **k: (state, comparison, submission),
    )
    monkeypatch.setattr(
        resume_model_release, "extract_adapter_checkpoint", lambda *a, **k: None
    )
    monkeypatch.setattr(
        resume_model_release,
        "resolve_evaluated_adapter_base_model",
        lambda *a, **k: "base/model",
    )
    monkeypatch.setattr(
        resume_model_release,
        "verify_selected_release_artifacts",
        lambda **k: None,
    )
    monkeypatch.setattr(
        resume_model_release,
        "audit_existing_model_bundle",
        lambda *a, **k: digest,
    )
    monkeypatch.setattr(
        resume_model_release,
        "verify_huggingface_adapter",
        lambda *a, **k: digest,
    )
    monkeypatch.setattr(
        resume_model_release, "verify_kaggle_adapter", lambda *a, **k: digest
    )
    monkeypatch.setattr(
        resume_model_release,
        "verify_anonymous_public_page",
        lambda url, **k: calls.append(("public", url)),
    )
    monkeypatch.setattr(
        resume_model_release,
        "publish_huggingface_space",
        lambda *a, **k: calls.append(("stage-space", k["private"])),
    )
    monkeypatch.setattr(
        resume_model_release,
        "set_huggingface_repo_visibility",
        lambda repo_id, **k: calls.append(("hf-visibility", repo_id, k)),
    )
    monkeypatch.setattr(
        resume_model_release, "update_private_manifest", lambda *a, **k: None
    )
    monkeypatch.setattr(
        resume_model_release,
        "write_selected_release_record",
        lambda *a, **k: {"kaggle_model": {"source": "source"}},
    )

    resume_model_release.main()

    assert calls[0][0] == "public"
    assert "kaggle.com/models" in calls[0][1]
    assert calls[1] == ("stage-space", True)
    assert calls[2][0] == "hf-visibility"
    assert calls[2][2] == {"repo_type": "model", "private": False}


def test_resume_existing_record_stops_before_remote_calls(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    args = _resume_args(tmp_path)
    args.selected_release_record.write_text("{}\n", encoding="utf-8")
    monkeypatch.setattr(resume_model_release, "load_dotenv", lambda: True)
    monkeypatch.setattr(resume_model_release, "parse_args", lambda: args)
    monkeypatch.setattr(
        resume_model_release,
        "await_passing_evaluation",
        lambda *a, **k: pytest.fail("remote preflight must not run"),
    )

    with pytest.raises(ValueError, match="already exists"):
        resume_model_release.main()
