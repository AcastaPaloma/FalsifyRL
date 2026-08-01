from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from falsifyrl.autoscientist import WorkflowState
from falsifyrl.dataset import DatasetBuildConfig, build_cases
from falsifyrl.evaluation import evaluate_completions, load_prediction_jsonl
from falsifyrl.reporting import build_comparison_report


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def evaluate_exact_predictions(path: Path) -> dict:
    cases = [
        case
        for case in build_cases(DatasetBuildConfig())
        if case.scenario.split.value == "test"
    ]
    completions = load_prediction_jsonl(path)
    expected_ids = {case.example_id for case in cases}
    actual_ids = set(completions)
    if actual_ids != expected_ids:
        missing = len(expected_ids - actual_ids)
        unexpected = len(actual_ids - expected_ids)
        raise ValueError(
            "predictions must cover the exact held-out split: "
            f"missing={missing}, unexpected={unexpected}"
        )
    return {
        "split": "test",
        "source": str(path.resolve()),
        "predictions_sha256": _sha256(path),
        "metrics": evaluate_completions(cases, completions).to_dict(),
    }


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _update_private_manifest(
    path: Path,
    state: WorkflowState,
    base_report: dict,
    adapted_report: dict,
) -> None:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    manifest["identifiers"]["autoscientist_run_id"] = state.autoscientist_run_id
    manifest["identifiers"]["base_model_id"] = state.resolved_model
    manifest["metrics"]["autoscientist_best_win_rate"] = state.best_win_rate
    manifest["metrics"]["base_model_composite"] = base_report["metrics"][
        "composite_score"
    ]
    manifest["metrics"]["trained_model_composite"] = adapted_report["metrics"][
        "composite_score"
    ]
    manifest["metrics"]["trained_json_validity"] = adapted_report["metrics"][
        "json_validity"
    ]
    temporary = path.with_suffix(".json.tmp")
    _write_json(temporary, manifest)
    temporary.replace(path)


def finalize(
    *,
    state_path: Path,
    base_predictions: Path,
    adapted_predictions: Path,
    adapter_weights: Path,
    dataset_manifest: Path,
    base_report_path: Path,
    adapted_report_path: Path,
    comparison_json_path: Path,
    comparison_markdown_path: Path,
    submission_manifest_path: Path | None,
) -> dict:
    state = WorkflowState.load(state_path)
    if (
        state.autoscientist_status != "succeeded"
        or not state.autoscientist_run_id
        or not state.download_available
        or state.best_win_rate is None
        or not state.resolved_model
    ):
        raise ValueError("a completed downloadable AutoScientist run is required")
    if not adapter_weights.is_file():
        raise FileNotFoundError(adapter_weights)
    if not dataset_manifest.is_file():
        raise FileNotFoundError(dataset_manifest)

    base_report = evaluate_exact_predictions(base_predictions)
    adapted_report = evaluate_exact_predictions(adapted_predictions)
    comparison = build_comparison_report(
        base_report,
        adapted_report,
        base_model_id=state.resolved_model,
        dataset_manifest_sha256=_sha256(dataset_manifest),
        adapter_sha256=_sha256(adapter_weights),
        autoscientist_run_id=state.autoscientist_run_id,
    )
    _write_json(base_report_path, base_report)
    _write_json(adapted_report_path, adapted_report)
    _write_json(comparison_json_path, comparison.value)
    comparison_markdown_path.parent.mkdir(parents=True, exist_ok=True)
    comparison_markdown_path.write_text(comparison.to_markdown(), encoding="utf-8")
    if submission_manifest_path is not None:
        _update_private_manifest(
            submission_manifest_path,
            state,
            base_report,
            adapted_report,
        )
    return comparison.value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Finalize exact CPU-side FalsifyRL metrics from GPU-generated Colab "
            "base and adapter predictions."
        )
    )
    parser.add_argument(
        "--state",
        type=Path,
        default=Path("outputs/autoscientist/workflow.json"),
    )
    parser.add_argument("--base-predictions", type=Path, required=True)
    parser.add_argument("--adapted-predictions", type=Path, required=True)
    parser.add_argument("--adapter-weights", type=Path, required=True)
    parser.add_argument(
        "--dataset-manifest",
        type=Path,
        default=Path("artifacts/release/adapted-dataset/release-manifest.json"),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--submission-manifest",
        type=Path,
        default=Path("outputs/submission/manifest.json"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    value = finalize(
        state_path=args.state,
        base_predictions=args.base_predictions,
        adapted_predictions=args.adapted_predictions,
        adapter_weights=args.adapter_weights,
        dataset_manifest=args.dataset_manifest,
        base_report_path=args.output_dir / "base-test.json",
        adapted_report_path=args.output_dir / "model-test.json",
        comparison_json_path=args.output_dir / "comparison.json",
        comparison_markdown_path=args.output_dir / "comparison.md",
        submission_manifest_path=args.submission_manifest,
    )
    print(json.dumps(value, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
