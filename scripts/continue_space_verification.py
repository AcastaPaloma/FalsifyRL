from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import requests
from dotenv import load_dotenv

from falsifyrl.schema import Diagnosis


def await_published_space(
    manifest_path: Path,
    examples_path: Path,
    *,
    poll_seconds: float,
    timeout_seconds: float,
) -> tuple[str, list[dict[str, Any]]]:
    deadline = time.monotonic() + timeout_seconds
    while True:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        space_url = manifest["links"].get("huggingface_space")
        if (
            space_url
            and manifest["links"].get("huggingface_model")
            and manifest["attestations"].get("weights_public_on_both_platforms")
            is True
            and examples_path.is_file()
        ):
            examples = json.loads(examples_path.read_text(encoding="utf-8"))
            return str(space_url), examples
        if time.monotonic() >= deadline:
            raise TimeoutError("published Space did not become available")
        time.sleep(poll_seconds)


def require_strict_prediction(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise TypeError("Space prediction must be a JSON object")
    if "error" in value or value.get("status") == "checkpoint_pending":
        raise RuntimeError(f"Space returned a non-model response: {value}")
    diagnosis = Diagnosis.from_json(json.dumps(value))
    return json.loads(diagnosis.to_json())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Wait for the public FalsifyRL Space, call its anonymous API on "
            "held-out control and exploit traces, and require strict diagnoses."
        )
    )
    parser.add_argument(
        "--submission-manifest",
        type=Path,
        default=Path("outputs/submission/manifest.json"),
    )
    parser.add_argument(
        "--examples",
        type=Path,
        default=Path("artifacts/release/space/examples.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/space-verification.json"),
    )
    parser.add_argument("--poll-seconds", type=float, default=30.0)
    parser.add_argument("--timeout-seconds", type=float, default=259_200.0)
    parser.add_argument("--prediction-timeout-seconds", type=float, default=900.0)
    return parser.parse_args()


def main() -> None:
    load_dotenv()
    args = parse_args()
    space_url, examples = await_published_space(
        args.submission_manifest,
        args.examples,
        poll_seconds=args.poll_seconds,
        timeout_seconds=args.timeout_seconds,
    )
    parts = space_url.rstrip("/").split("/")
    if len(parts) < 2:
        raise ValueError(f"invalid Hugging Face Space URL: {space_url}")
    repo_id = "/".join(parts[-2:])

    from huggingface_hub import HfApi

    api = HfApi()
    deadline = time.monotonic() + args.timeout_seconds
    while True:
        runtime = api.get_space_runtime(repo_id, token=False)
        stage = str(runtime.stage).upper()
        if stage.endswith("RUNNING"):
            break
        if any(
            stage.endswith(terminal)
            for terminal in ("BUILD_ERROR", "CONFIG_ERROR", "RUNTIME_ERROR")
        ):
            raise RuntimeError(f"Space entered terminal stage {runtime.stage}")
        if time.monotonic() >= deadline:
            raise TimeoutError(f"Space did not reach RUNNING; latest stage: {runtime.stage}")
        time.sleep(args.poll_seconds)

    selected = {}
    for role in ("control", "exploit"):
        selected[role] = next(
            example for example in examples if example["case_role"] == role
        )
    results = {}
    info = api.space_info(repo_id, token=False)
    filenames = {sibling.rfilename for sibling in info.siblings or []}
    if "index.html" in filenames:
        owner, slug = repo_id.split("/", maxsplit=1)
        app_url = f"https://{owner.lower()}-{slug.lower()}.static.hf.space"
        page = requests.get(f"{app_url}/", timeout=args.prediction_timeout_seconds)
        page.raise_for_status()
        if "FalsifyRL" not in page.text:
            raise RuntimeError("static Space page is missing the FalsifyRL marker")
        remote_examples_response = requests.get(
            f"{app_url}/examples.json",
            timeout=args.prediction_timeout_seconds,
        )
        remote_examples_response.raise_for_status()
        remote_examples = remote_examples_response.json()
        if {
            str(example["example_id"]) for example in remote_examples
        } != {str(example["example_id"]) for example in examples}:
            raise RuntimeError("static Space examples differ from the release bundle")
        predictions_response = requests.get(
            f"{app_url}/predictions.json",
            timeout=args.prediction_timeout_seconds,
        )
        predictions_response.raise_for_status()
        prediction_bundle = predictions_response.json()
        if (
            prediction_bundle.get("schema_version") != 1
            or prediction_bundle.get("mode")
            != "cached_exact_checkpoint_predictions"
            or len(str(prediction_bundle.get("source_predictions_sha256", "")))
            != 64
        ):
            raise RuntimeError("static Space prediction provenance is invalid")
        remote_predictions = prediction_bundle.get("predictions", {})
        for role, example in selected.items():
            results[role] = {
                "example_id": example["example_id"],
                "prediction": require_strict_prediction(
                    remote_predictions.get(example["example_id"])
                ),
            }
        value = {
            "space_url": space_url,
            "public_app_url": app_url,
            "space_stage": "RUNNING",
            "anonymous_interface": "static cached evidence explorer",
            "source_predictions_sha256": prediction_bundle[
                "source_predictions_sha256"
            ],
            "roles_verified": sorted(results),
            "results": results,
        }
    else:
        from gradio_client import Client

        client = Client(space_url, verbose=False)
        for role, example in selected.items():
            job = client.submit(
                example["example_id"],
                api_name="/run_critic",
            )
            prediction = job.result(timeout=args.prediction_timeout_seconds)
            results[role] = {
                "example_id": example["example_id"],
                "prediction": require_strict_prediction(prediction),
            }
        value = {
            "space_url": space_url,
            "space_stage": "RUNNING",
            "anonymous_api": "/run_critic",
            "roles_verified": sorted(results),
            "results": results,
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(value, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
