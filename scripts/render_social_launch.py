from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

REPLACEMENTS = {
    "AUTOSCIENTIST_BEST_WIN_RATE": ("metrics", "autoscientist_best_win_rate"),
    "BASE_COMPOSITE_SCORE": ("metrics", "base_model_composite"),
    "TRAINED_COMPOSITE_SCORE": ("metrics", "trained_model_composite"),
    "TRAINED_JSON_VALIDITY": ("metrics", "trained_json_validity"),
    "GITHUB_URL": ("links", "github"),
    "HUGGINGFACE_MODEL_URL": ("links", "huggingface_model"),
    "HUGGINGFACE_SPACE_URL": ("links", "huggingface_space"),
    "KAGGLE_NOTEBOOK_URL": ("links", "kaggle_notebook"),
    "EVALUATION_REPORT_URL": ("links", "evaluation_report"),
}


def render_social_launch(template: Path, manifest_path: Path, output: Path) -> Path:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    values: dict[str, str] = {}
    for marker, (section, key) in REPLACEMENTS.items():
        value = manifest.get(section, {}).get(key)
        if value is None or value == "":
            raise ValueError(f"submission manifest is missing {section}.{key}")
        if marker == "TRAINED_JSON_VALIDITY":
            values[marker] = f"{float(value):.2%}"
        elif section == "metrics":
            values[marker] = f"{float(value):.4f}"
        else:
            values[marker] = str(value)

    content = template.read_text(encoding="utf-8")
    for marker, value in values.items():
        content = content.replace("{{" + marker + "}}", value)
    unresolved = sorted(set(re.findall(r"\{\{[A-Z0-9_]+\}\}", content)))
    if unresolved:
        raise ValueError(f"social launch draft has unresolved placeholders: {unresolved}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(content, encoding="utf-8")
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render approval-ready social drafts from the audited private manifest."
    )
    parser.add_argument(
        "--template",
        type=Path,
        default=Path("docs/social_launch_draft.md"),
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("outputs/submission/manifest.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/submission/social_launch_ready.md"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = render_social_launch(args.template, args.manifest, args.output)
    print(output.resolve())


if __name__ == "__main__":
    main()
