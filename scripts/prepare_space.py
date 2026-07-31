from __future__ import annotations

import argparse
import json
from pathlib import Path

from falsifyrl.demo import prepare_space_bundle


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare the FalsifyRL Hugging Face Space bundle.")
    parser.add_argument("--template-dir", type=Path, default=Path("space"))
    parser.add_argument(
        "--dataset-jsonl",
        type=Path,
        default=Path("outputs/falsifyrl_seed_v1/test.jsonl"),
    )
    parser.add_argument(
        "--bundle-dir",
        type=Path,
        default=Path("artifacts/release/space"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = prepare_space_bundle(
        args.template_dir,
        args.dataset_jsonl,
        args.bundle_dir,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

