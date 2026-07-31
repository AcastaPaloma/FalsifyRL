from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from falsifyrl.reporting import build_comparison_report


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a fail-closed held-out base-versus-adapter report."
    )
    parser.add_argument("--base-report", type=Path, required=True)
    parser.add_argument("--adapted-report", type=Path, required=True)
    parser.add_argument("--base-model-id", required=True)
    parser.add_argument("--dataset-manifest", type=Path, required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--autoscientist-run-id", required=True)
    parser.add_argument(
        "--json-output",
        type=Path,
        default=Path("outputs/evaluation/comparison.json"),
    )
    parser.add_argument(
        "--markdown-output",
        type=Path,
        default=Path("outputs/evaluation/comparison.md"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = build_comparison_report(
        json.loads(args.base_report.read_text(encoding="utf-8")),
        json.loads(args.adapted_report.read_text(encoding="utf-8")),
        base_model_id=args.base_model_id,
        dataset_manifest_sha256=_sha256(args.dataset_manifest),
        adapter_sha256=_sha256(args.adapter),
        autoscientist_run_id=args.autoscientist_run_id,
    )
    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.markdown_output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.write_text(
        json.dumps(report.value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    args.markdown_output.write_text(report.to_markdown(), encoding="utf-8")
    print(json.dumps(report.value, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
