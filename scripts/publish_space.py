from __future__ import annotations

import argparse
import json
from pathlib import Path

from falsifyrl.release import publish_huggingface_space


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Publish the trained FalsifyRL Gradio Space.")
    parser.add_argument("--owner", required=True)
    parser.add_argument("--base-model-id", required=True)
    parser.add_argument("--model-repo-id", required=True)
    parser.add_argument("--slug", default="falsifyrl")
    parser.add_argument(
        "--bundle-dir",
        type=Path,
        default=Path("artifacts/release/space"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    url = publish_huggingface_space(
        args.bundle_dir,
        owner=args.owner,
        base_model_id=args.base_model_id,
        model_repo_id=args.model_repo_id,
        slug=args.slug,
    )
    print(json.dumps({"platform": "huggingface-space", "url": url}, indent=2))


if __name__ == "__main__":
    main()

