from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from dotenv import load_dotenv

from falsifyrl.release import publish_huggingface_dataset, publish_kaggle_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Publish the audited FalsifyRL dataset bundle.")
    parser.add_argument("platform", choices=("huggingface", "kaggle"))
    parser.add_argument("--owner")
    parser.add_argument("--slug", default="falsifyrl-seed")
    parser.add_argument(
        "--bundle-dir",
        type=Path,
        default=Path("artifacts/release/dataset"),
    )
    return parser.parse_args()


def main() -> None:
    load_dotenv()
    args = parse_args()
    owner = args.owner or os.environ.get(
        "FALSIFYRL_HF_OWNER"
        if args.platform == "huggingface"
        else "FALSIFYRL_KAGGLE_OWNER"
    )
    if not owner:
        raise RuntimeError(f"owner is required for {args.platform}")
    if args.platform == "huggingface":
        url = publish_huggingface_dataset(
            args.bundle_dir,
            owner=owner,
            slug=args.slug,
        )
    else:
        url = publish_kaggle_dataset(
            args.bundle_dir,
            owner=owner,
            slug=args.slug,
        )
    print(json.dumps({"platform": args.platform, "url": url}, indent=2))


if __name__ == "__main__":
    main()
