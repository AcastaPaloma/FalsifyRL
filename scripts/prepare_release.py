from __future__ import annotations

import argparse
import json
from pathlib import Path

from falsifyrl.release import prepare_dataset_bundle


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare audited public FalsifyRL release bundles."
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=Path("outputs/falsifyrl_seed_v1"),
    )
    parser.add_argument(
        "--bundle-dir",
        type=Path,
        default=Path("artifacts/release/dataset"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = prepare_dataset_bundle(args.dataset_dir, args.bundle_dir)
    print(
        json.dumps(
            {
                "bundle_dir": str(args.bundle_dir.resolve()),
                "case_count": manifest["case_count"],
                "files": sorted(manifest["files"]),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
