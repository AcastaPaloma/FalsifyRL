from __future__ import annotations

import argparse
import json
from pathlib import Path

from falsifyrl.dataset import DatasetBuildConfig, write_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate and verify the deterministic FalsifyRL seed dataset."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/falsifyrl_seed_v1"),
    )
    parser.add_argument("--train-seeds", type=int, default=80)
    parser.add_argument("--validation-seeds", type=int, default=40)
    parser.add_argument("--test-seeds", type=int, default=40)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = DatasetBuildConfig(
        train_seed_count=args.train_seeds,
        validation_seed_count=args.validation_seeds,
        test_seed_count=args.test_seeds,
    )
    manifest = write_dataset(args.output_dir, config=config)
    print(
        json.dumps(
            {
                "output_dir": str(args.output_dir.resolve()),
                "case_count": manifest["validation"]["case_count"],
                "pair_count": manifest["validation"]["pair_count"],
                "split_counts": manifest["validation"]["split_counts"],
                "all_cases_verified": manifest["validation"]["all_cases_verified"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()

