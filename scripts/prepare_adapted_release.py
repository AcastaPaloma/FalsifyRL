from __future__ import annotations

import argparse
import json
from pathlib import Path

from falsifyrl.release import prepare_adapted_dataset_bundle


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare the exact audited AutoScientist-adapted dataset bundle."
    )
    parser.add_argument(
        "--seed-dataset-dir",
        type=Path,
        default=Path("outputs/falsifyrl_seed_v1"),
    )
    parser.add_argument(
        "--adapted-csv",
        type=Path,
        default=Path("outputs/autoscientist/adapted-train.csv"),
    )
    parser.add_argument(
        "--adaptation-audit",
        type=Path,
        default=Path("outputs/autoscientist/adapted-train.audit.json"),
    )
    parser.add_argument(
        "--bundle-dir",
        type=Path,
        default=Path("artifacts/release/adapted-dataset"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = prepare_adapted_dataset_bundle(
        args.seed_dataset_dir,
        args.adapted_csv,
        args.adaptation_audit,
        args.bundle_dir,
    )
    print(
        json.dumps(
            {
                "bundle_dir": str(args.bundle_dir.resolve()),
                "dataset_variant": manifest["dataset_variant"],
                "training_file_sha256": manifest["training_file_sha256"],
                "files": sorted(manifest["files"]),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
