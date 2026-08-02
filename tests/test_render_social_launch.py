from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.render_social_launch import render_social_launch


def _manifest() -> dict:
    return {
        "metrics": {
            "autoscientist_best_win_rate": 0.9643,
            "base_model_composite": 0.0,
            "trained_model_composite": 0.902160771,
            "trained_json_validity": 0.9828125,
        },
        "links": {
            "github": "https://github.com/owner/repo",
            "huggingface_model": "https://huggingface.co/owner/model",
            "huggingface_space": "https://huggingface.co/spaces/owner/demo",
            "kaggle_notebook": "https://www.kaggle.com/code/owner/eval",
            "evaluation_report": "https://huggingface.co/owner/model/report",
        },
    }


def test_social_launch_is_rendered_only_from_audited_values(tmp_path: Path) -> None:
    template = tmp_path / "template.md"
    manifest = tmp_path / "manifest.json"
    output = tmp_path / "ready.md"
    template.write_text(
        "{{AUTOSCIENTIST_BEST_WIN_RATE}} {{BASE_COMPOSITE_SCORE}} "
        "{{TRAINED_COMPOSITE_SCORE}} {{TRAINED_JSON_VALIDITY}} "
        "{{GITHUB_URL}} {{HUGGINGFACE_MODEL_URL}} {{HUGGINGFACE_SPACE_URL}} "
        "{{KAGGLE_NOTEBOOK_URL}} {{EVALUATION_REPORT_URL}}",
        encoding="utf-8",
    )
    manifest.write_text(json.dumps(_manifest()), encoding="utf-8")

    render_social_launch(template, manifest, output)

    content = output.read_text(encoding="utf-8")
    assert "0.9643 0.0000 0.9022 98.28%" in content
    assert "{{" not in content


def test_social_launch_fails_closed_until_every_public_link_exists(
    tmp_path: Path,
) -> None:
    template = tmp_path / "template.md"
    manifest = tmp_path / "manifest.json"
    template.write_text("{{KAGGLE_NOTEBOOK_URL}}", encoding="utf-8")
    value = _manifest()
    value["links"]["kaggle_notebook"] = None
    manifest.write_text(json.dumps(value), encoding="utf-8")

    with pytest.raises(ValueError, match="links.kaggle_notebook"):
        render_social_launch(template, manifest, tmp_path / "ready.md")
