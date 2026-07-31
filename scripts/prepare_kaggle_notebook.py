from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path
from typing import Any

from dotenv import load_dotenv


def prepare_kaggle_notebook(
    *,
    owner: str,
    notebook_path: Path,
    metadata_template: Path,
    output_dir: Path,
    model_version: int = 1,
) -> dict[str, Any]:
    if not owner.strip() or owner == "OWNER":
        raise ValueError("a real Kaggle owner is required")
    if model_version < 1:
        raise ValueError("model version must be positive")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError(f"output directory must be empty: {output_dir}")

    metadata = json.loads(metadata_template.read_text(encoding="utf-8"))
    metadata["id"] = f"{owner}/falsifyrl-held-out-evaluation"
    metadata["dataset_sources"] = [f"{owner}/falsifyrl-adapted"]
    metadata["model_sources"] = [
        f"{owner}/falsifyrl-autoscientist/pytorch/lora/{model_version}"
    ]

    output_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(notebook_path, output_dir / notebook_path.name)
    (output_dir / "kernel-metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stage the public Kaggle evaluation notebook with exact data/model inputs."
    )
    parser.add_argument("--owner")
    parser.add_argument("--model-version", type=int, default=1)
    parser.add_argument(
        "--notebook",
        type=Path,
        default=Path("kaggle/falsifyrl_evaluation.ipynb"),
    )
    parser.add_argument(
        "--metadata-template",
        type=Path,
        default=Path("kaggle/kernel-metadata.template.json"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/release/kaggle-notebook"),
    )
    return parser.parse_args()


def main() -> None:
    load_dotenv()
    args = parse_args()
    owner = args.owner or os.environ.get("FALSIFYRL_KAGGLE_OWNER")
    if not owner:
        raise RuntimeError("Kaggle owner is required")
    metadata = prepare_kaggle_notebook(
        owner=owner,
        notebook_path=args.notebook,
        metadata_template=args.metadata_template,
        output_dir=args.output_dir,
        model_version=args.model_version,
    )
    print(json.dumps(metadata, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
