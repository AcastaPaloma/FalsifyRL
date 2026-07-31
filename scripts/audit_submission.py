from __future__ import annotations

import argparse
import json
from pathlib import Path

from falsifyrl.submission import audit_submission_manifest, load_submission_manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fail closed unless every FalsifyRL submission requirement is present."
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("outputs/submission/manifest.json"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    audit = audit_submission_manifest(load_submission_manifest(args.manifest))
    print(json.dumps(audit.to_dict(), indent=2, sort_keys=True))
    if not audit.valid:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

