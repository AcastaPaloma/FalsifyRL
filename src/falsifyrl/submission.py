from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

SUBMISSION_FORM_URL = "https://share.hsforms.com/2xleXmJ7wSkimSzP8L55KcAuc9yb"
REQUIRED_LINKS = {
    "github": ("github.com",),
    "huggingface_dataset": ("huggingface.co",),
    "kaggle_dataset": ("kaggle.com", "www.kaggle.com"),
    "huggingface_model": ("huggingface.co",),
    "kaggle_model": ("kaggle.com", "www.kaggle.com"),
    "huggingface_space": ("huggingface.co",),
    "kaggle_notebook": ("kaggle.com", "www.kaggle.com"),
    "evaluation_report": ("github.com", "huggingface.co", "kaggle.com"),
}
REQUIRED_IDS = {
    "adaption_dataset_id",
    "autoscientist_run_id",
    "base_model_id",
}
REQUIRED_FORM_INPUTS = {
    "first_name",
    "last_name",
    "email",
    "job_title",
    "company_name",
    "street_address",
    "city",
    "state_region",
    "postal_code",
    "country",
    "discord_username",
}
REQUIRED_ATTESTATIONS = {
    "accepted_into_challenge",
    "at_least_18",
    "not_quebec_resident",
    "participation_legal",
    "one_team_only",
    "terms_and_conditions_accepted",
    "dataset_public_on_both_platforms",
    "same_dataset_used_for_training",
    "weights_public_on_both_platforms",
    "heldout_family_never_trained",
    "all_patches_executable",
    "no_secrets_in_public_artifacts",
}


@dataclass(frozen=True)
class SubmissionAudit:
    valid: bool
    errors: tuple[str, ...]
    warnings: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "valid": self.valid,
            "error_count": len(self.errors),
            "warning_count": len(self.warnings),
            "errors": list(self.errors),
            "warnings": list(self.warnings),
        }


def _valid_public_url(value: Any, allowed_hosts: tuple[str, ...]) -> bool:
    if not isinstance(value, str) or not value:
        return False
    parsed = urlparse(value)
    return parsed.scheme == "https" and parsed.hostname in allowed_hosts


def audit_submission_manifest(manifest: dict[str, Any]) -> SubmissionAudit:
    errors: list[str] = []
    warnings: list[str] = []
    if manifest.get("challenge") != "Adaption AutoScientist Challenge Part 2":
        errors.append("challenge name is missing or incorrect")
    if manifest.get("category") != "Science":
        errors.append("submission category must be Science")
    if manifest.get("submission_form_url") != SUBMISSION_FORM_URL:
        errors.append("official submission form URL is missing or incorrect")

    links = manifest.get("links", {})
    for name, hosts in REQUIRED_LINKS.items():
        if not _valid_public_url(links.get(name), hosts):
            errors.append(f"required public link is missing or invalid: {name}")
    for optional_social in ("linkedin_post", "x_post"):
        value = links.get(optional_social)
        if not value:
            warnings.append(f"bonus social link is missing: {optional_social}")
        elif not _valid_public_url(
            value,
            (
                "linkedin.com",
                "www.linkedin.com",
                "x.com",
                "www.x.com",
                "twitter.com",
            ),
        ):
            errors.append(f"social link is invalid: {optional_social}")

    identifiers = manifest.get("identifiers", {})
    for name in REQUIRED_IDS:
        value = identifiers.get(name)
        if not isinstance(value, str) or not value.strip():
            errors.append(f"required identifier is missing: {name}")

    form_inputs = manifest.get("form_inputs", {})
    for name in REQUIRED_FORM_INPUTS:
        value = form_inputs.get(name)
        if not isinstance(value, str) or not value.strip():
            errors.append(f"required private form input is missing: {name}")
    if not isinstance(form_inputs.get("hackindia_submission"), bool):
        errors.append("HackIndia submission status must be explicitly true or false")

    attestations = manifest.get("attestations", {})
    for name in REQUIRED_ATTESTATIONS:
        if attestations.get(name) is not True:
            errors.append(f"required attestation is not true: {name}")

    metrics = manifest.get("metrics", {})
    base_composite = metrics.get("base_model_composite")
    trained_composite = metrics.get("trained_model_composite")
    best_win_rate = metrics.get("autoscientist_best_win_rate")
    json_validity = metrics.get("trained_json_validity")
    if not isinstance(base_composite, (float, int)):
        errors.append("base_model_composite is missing")
    if not isinstance(trained_composite, (float, int)):
        errors.append("trained_model_composite is missing")
    if (
        isinstance(base_composite, (float, int))
        and isinstance(trained_composite, (float, int))
        and trained_composite <= base_composite
    ):
        errors.append("trained model does not improve over the base model")
    if not isinstance(best_win_rate, (float, int)) or best_win_rate <= 0.5:
        errors.append("AutoScientist best win rate must exceed 0.5")
    if not isinstance(json_validity, (float, int)) or json_validity < 0.95:
        errors.append("trained JSON validity must be at least 0.95")

    dataset = manifest.get("dataset", {})
    if dataset.get("case_count") != 3840:
        errors.append("dataset case count must match the verified v1 release")
    if dataset.get("test_family") != "crossing_navigation":
        errors.append("held-out test family must be crossing_navigation")
    if dataset.get("variant") not in {"adapted", "passthrough_verified"}:
        errors.append("dataset variant must identify the exact trainable release")
    if not dataset.get("sha256_manifest"):
        errors.append("dataset SHA-256 manifest reference is missing")

    return SubmissionAudit(
        valid=not errors,
        errors=tuple(errors),
        warnings=tuple(warnings),
    )


def load_submission_manifest(path: str | Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("submission manifest root must be an object")
    return value
