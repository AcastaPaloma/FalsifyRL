from __future__ import annotations

import copy
import json
from pathlib import Path

from falsifyrl.submission import audit_submission_manifest, load_submission_manifest


def _valid_manifest() -> dict:
    return {
        "challenge": "Adaption AutoScientist Challenge Part 2",
        "category": "Science",
        "submission_form_url": (
            "https://share.hsforms.com/2xleXmJ7wSkimSzP8L55KcAuc9yb"
        ),
        "dataset": {
            "case_count": 3840,
            "pair_count": 1920,
            "variant": "adapted",
            "test_family": "crossing_navigation",
            "sha256_manifest": "https://huggingface.co/manifest.json",
        },
        "identifiers": {
            "adaption_dataset_id": "dataset-1",
            "autoscientist_run_id": "run-1",
            "base_model_id": "org/base",
        },
        "form_inputs": {
            "first_name": "Ada",
            "last_name": "Lovelace",
            "email": "ada@example.test",
            "job_title": "Researcher",
            "company_name": "Independent",
            "street_address": "1 Example Street",
            "city": "Example City",
            "state_region": "Example Region",
            "postal_code": "A1A 1A1",
            "country": "Canada",
            "discord_username": "ada",
            "hackindia_submission": False,
        },
        "metrics": {
            "base_model_composite": 0.45,
            "trained_model_composite": 0.8,
            "autoscientist_best_win_rate": 0.81,
            "trained_json_validity": 0.99,
        },
        "links": {
            "github": "https://github.com/owner/falsifyrl",
            "huggingface_dataset": "https://huggingface.co/datasets/owner/data",
            "kaggle_dataset": "https://www.kaggle.com/datasets/owner/data",
            "huggingface_model": "https://huggingface.co/owner/model",
            "kaggle_model": "https://www.kaggle.com/models/owner/model",
            "huggingface_space": "https://huggingface.co/spaces/owner/demo",
            "kaggle_notebook": "https://www.kaggle.com/code/owner/eval",
            "evaluation_report": "https://github.com/owner/falsifyrl/report.json",
            "linkedin_post": None,
            "x_post": None,
        },
        "attestations": {
            "accepted_into_challenge": True,
            "at_least_18": True,
            "not_quebec_resident": True,
            "participation_legal": True,
            "one_team_only": True,
            "terms_and_conditions_accepted": True,
            "dataset_public_on_both_platforms": True,
            "same_dataset_used_for_training": True,
            "weights_public_on_both_platforms": True,
            "heldout_family_never_trained": True,
            "all_patches_executable": True,
            "no_secrets_in_public_artifacts": True,
        },
    }


def test_complete_submission_passes_with_only_bonus_warnings() -> None:
    audit = audit_submission_manifest(_valid_manifest())

    assert audit.valid is True
    assert audit.errors == ()
    assert len(audit.warnings) == 2


def test_submission_fails_closed_on_missing_artifact_or_improvement() -> None:
    manifest = copy.deepcopy(_valid_manifest())
    manifest["links"]["kaggle_model"] = None
    manifest["metrics"]["trained_model_composite"] = 0.4
    manifest["form_inputs"]["hackindia_submission"] = None

    audit = audit_submission_manifest(manifest)

    assert audit.valid is False
    assert any("kaggle_model" in error for error in audit.errors)
    assert any("does not improve" in error for error in audit.errors)
    assert any("HackIndia" in error for error in audit.errors)


def test_submission_template_is_intentionally_incomplete() -> None:
    template_path = Path("submission/manifest.template.json")
    template = load_submission_manifest(template_path)

    audit = audit_submission_manifest(template)

    assert audit.valid is False
    assert len(audit.errors) >= 10


def test_manifest_loader_rejects_non_object(tmp_path: Path) -> None:
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps([]), encoding="utf-8")

    try:
        load_submission_manifest(path)
    except ValueError as error:
        assert "root must be an object" in str(error)
    else:
        raise AssertionError("expected non-object manifest rejection")
