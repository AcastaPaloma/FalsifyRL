from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from falsifyrl.release import (
    DATASET_FILES,
    audit_model_bundle,
    prepare_adapted_dataset_bundle,
    prepare_dataset_bundle,
    render_model_card,
    require_huggingface_token,
    require_kaggle_token,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fake_dataset(source: Path) -> None:
    source.mkdir()
    file_manifest = {}
    for filename in DATASET_FILES:
        if filename == "manifest.json":
            continue
        path = source / filename
        path.write_text(f"{filename}\n", encoding="utf-8")
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
