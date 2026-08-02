from __future__ import annotations

import csv
import hashlib
import json
import os
import shutil
import tarfile
import tempfile
from pathlib import Path
from typing import Any

from falsifyrl.schema import Diagnosis

DATASET_FILES = (
    "train.csv",
    "train.jsonl",
    "validation.csv",
    "validation.jsonl",
    "test.csv",
    "test.jsonl",
    "manifest.json",
)

ZSTANDARD_MAGIC = b"\x28\xb5\x2f\xfd"
MAX_DECOMPRESSED_CHECKPOINT_BYTES = 4 * 1024 * 1024 * 1024

ADAPTED_DATASET_SUPPORT_FILES = (
    "validation.csv",
    "validation.jsonl",
    "test.csv",
    "test.jsonl",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_anonymous_public_page(
    url: str,
    *,
    expected_marker: str,
    fetcher: Any | None = None,
) -> None:
    if fetcher is None:
        try:
            import requests
        except ImportError as error:
            raise RuntimeError(
                "Install release dependencies with `pip install -e .[release]`."
            ) from error
        fetcher = requests.get
    response = fetcher(url, timeout=30, allow_redirects=True)
    final_url = str(response.url)
    if response.status_code != 200:
        raise RuntimeError(f"public artifact page returned HTTP {response.status_code}: {url}")
    if "/login" in final_url.lower() or "/account/" in final_url.lower():
        raise RuntimeError(f"public artifact page redirected to authentication: {final_url}")
    if expected_marker.lower() not in response.text.lower():
        raise RuntimeError(
            f"public artifact page does not contain expected marker: {expected_marker}"
        )


def prepare_dataset_bundle(
    dataset_dir: str | Path,
    bundle_dir: str | Path,
    *,
    card_path: str | Path = "release/dataset/README.md",
    license_path: str | Path = "LICENSE",
) -> dict[str, Any]:
    source = Path(dataset_dir)
    destination = Path(bundle_dir)
    source_manifest = json.loads((source / "manifest.json").read_text(encoding="utf-8"))

    for filename, metadata in source_manifest["files"].items():
        actual = _sha256(source / filename)
        if actual != metadata["sha256"]:
            raise ValueError(
                f"source hash mismatch for {filename}: {actual} != {metadata['sha256']}"
            )

    destination.mkdir(parents=True, exist_ok=True)
    for filename in DATASET_FILES:
        source_path = source / filename
        if not source_path.is_file():
            raise FileNotFoundError(source_path)
        shutil.copy2(source_path, destination / filename)
    shutil.copy2(card_path, destination / "README.md")
    shutil.copy2(license_path, destination / "LICENSE")

    bundle_files = {
        path.name: {"bytes": path.stat().st_size, "sha256": _sha256(path)}
        for path in sorted(destination.iterdir())
        if path.is_file()
    }
    release_manifest = {
        "artifact": "falsifyrl-seed-v1",
        "visibility": "public",
        "dataset_version": source_manifest["dataset_version"],
        "case_count": source_manifest["validation"]["case_count"],
        "split_counts": source_manifest["validation"]["split_counts"],
        "files": bundle_files,
    }
    (destination / "release-manifest.json").write_text(
        json.dumps(release_manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return release_manifest


def prepare_adapted_dataset_bundle(
    seed_dataset_dir: str | Path,
    adapted_csv: str | Path,
    adaptation_audit: str | Path,
    bundle_dir: str | Path,
    *,
    card_path: str | Path = "release/adapted_dataset/README.md",
    license_path: str | Path = "LICENSE",
) -> dict[str, Any]:
    seed = Path(seed_dataset_dir)
    adapted = Path(adapted_csv)
    audit_path = Path(adaptation_audit)
    destination = Path(bundle_dir)
    seed_manifest = json.loads((seed / "manifest.json").read_text(encoding="utf-8"))
    audit = json.loads(audit_path.read_text(encoding="utf-8"))

    for filename, metadata in seed_manifest["files"].items():
        actual = _sha256(seed / filename)
        if actual != metadata["sha256"]:
            raise ValueError(
                f"source hash mismatch for {filename}: {actual} != {metadata['sha256']}"
            )
    required_audit_truths = (
        "all_source_prompts_matched",
        "all_completions_strict_json",
        "all_diagnosis_invariants_preserved",
    )
    if audit.get("dataset_variant") != "adapted" or not all(
        audit.get(field) is True for field in required_audit_truths
    ):
        raise ValueError("adaptation audit does not prove an exact verified adapted dataset")
    expected_source_rows = seed_manifest["validation"]["split_counts"]["train"]
    source_completions: dict[str, str] = {}
    actual_source_rows = 0
    with (seed / "train.csv").open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        for row in reader:
            actual_source_rows += 1
            prompt = row["prompt"]
            completion = row["completion"]
            previous = source_completions.get(prompt)
            if previous is not None and completion != previous:
                raise ValueError(
                    "source training CSV contains a prompt with conflicting completions"
                )
            source_completions[prompt] = completion
    expected_unique_rows = len(source_completions)
    if actual_source_rows != expected_source_rows:
        raise ValueError("source training row count does not match the seed manifest")
    if audit.get("source_row_count") != expected_source_rows:
        raise ValueError("audit source row count does not match the verified training split")
    if audit.get("source_unique_row_count") != expected_unique_rows:
        raise ValueError("audit unique source count does not match source_train.csv")
    if audit.get("row_count") != expected_unique_rows:
        raise ValueError(
            "adapted row count does not match the exact-deduplicated training split"
        )
    if audit.get("exact_duplicate_rows_collapsed") != (
        expected_source_rows - expected_unique_rows
    ):
        raise ValueError("audit exact-duplicate collapse count is inconsistent")
    if audit.get("source_sha256") != _sha256(seed / "train.csv"):
        raise ValueError("adaptation audit source hash does not match seed train.csv")
    if audit.get("adapted_sha256") != _sha256(adapted):
        raise ValueError("adapted CSV hash does not match the adaptation audit")
    if not audit.get("dataset_id") or not audit.get("adaptation_run_id"):
        raise ValueError("adaptation audit is missing Adaption dataset/run identifiers")

    destination.mkdir(parents=True, exist_ok=True)
    shutil.copy2(adapted, destination / "train.csv")
    shutil.copy2(seed / "train.csv", destination / "source_train.csv")
    for filename in ADAPTED_DATASET_SUPPORT_FILES:
        shutil.copy2(seed / filename, destination / filename)
    shutil.copy2(seed / "manifest.json", destination / "seed-manifest.json")
    shutil.copy2(audit_path, destination / "adaptation-audit.json")
    shutil.copy2(card_path, destination / "README.md")
    shutil.copy2(license_path, destination / "LICENSE")

    bundle_files = {
        path.name: {"bytes": path.stat().st_size, "sha256": _sha256(path)}
        for path in sorted(destination.iterdir())
        if path.is_file()
    }
    release_manifest = {
        "artifact": "falsifyrl-adapted-v1",
        "visibility": "public",
        "dataset_variant": "adapted",
        "dataset_version": seed_manifest["dataset_version"],
        "case_count": seed_manifest["validation"]["case_count"],
        "split_counts": seed_manifest["validation"]["split_counts"],
        "adaption_dataset_id": audit["dataset_id"],
        "adaptation_run_id": audit["adaptation_run_id"],
        "training_file": "train.csv",
        "training_file_sha256": audit["adapted_sha256"],
        "training_row_count": audit["row_count"],
        "source_training_row_count": audit["source_row_count"],
        "exact_duplicate_rows_collapsed": audit["exact_duplicate_rows_collapsed"],
        "files": bundle_files,
    }
    manifest_path = destination / "release-manifest.json"
    manifest_path.write_text(
        json.dumps(release_manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return release_manifest


def require_huggingface_token() -> str:
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if not token:
        raise RuntimeError("HF_TOKEN is required in the process environment")
    return token


def require_kaggle_token() -> str:
    token = os.environ.get("KAGGLE_API_TOKEN")
    if not token:
        raise RuntimeError("KAGGLE_API_TOKEN is required in the process environment")
    return token


def set_kaggle_dataset_public(
    *,
    owner: str,
    slug: str,
    title: str,
    subtitle: str,
    description: str,
    license_name: str = "MIT",
) -> None:
    try:
        from kagglehub.clients import build_kaggle_client
        from kagglesdk.datasets.types.dataset_api_service import (
            ApiUpdateDatasetMetadataRequest,
        )
        from kagglesdk.datasets.types.dataset_types import (
            DatasetSettings,
            SettingsLicense,
        )
    except ImportError as error:
        raise RuntimeError(
            "Install release dependencies with `pip install -e .[release]`."
        ) from error
    settings = DatasetSettings()
    settings.title = title
    settings.subtitle = subtitle
    settings.description = description
    settings.is_private = False
    license_value = SettingsLicense()
    license_value.name = license_name
    settings.licenses = [license_value]
    settings.expected_update_frequency = "never"

    request = ApiUpdateDatasetMetadataRequest()
    request.owner_slug = owner
    request.dataset_slug = slug
    request.settings = settings
    with build_kaggle_client() as client:
        response = client.datasets.dataset_api_client.update_dataset_metadata(request)
    if response.errors:
        raise RuntimeError(f"Kaggle dataset metadata update failed: {response.errors}")


def set_kaggle_model_visibility(
    *,
    owner: str,
    slug: str,
    title: str,
    subtitle: str,
    description: str,
    private: bool,
) -> None:
    try:
        from google.protobuf.field_mask_pb2 import FieldMask
        from kagglehub.clients import build_kaggle_client
        from kagglesdk.models.types.model_api_service import ApiUpdateModelRequest
    except ImportError as error:
        raise RuntimeError(
            "Install release dependencies with `pip install -e .[release]`."
        ) from error
    request = ApiUpdateModelRequest()
    request.owner_slug = owner
    request.model_slug = slug
    request.title = title
    request.subtitle = subtitle
    request.description = description
    request.is_private = private
    request.update_mask = FieldMask(
        paths=("title", "subtitle", "description", "is_private")
    )
    with build_kaggle_client() as client:
        response = client.models.model_api_client.update_model(request)
    if response.error:
        raise RuntimeError(f"Kaggle model metadata update failed: {response.error}")


def set_kaggle_model_public(
    *,
    owner: str,
    slug: str,
    title: str,
    subtitle: str,
    description: str,
) -> None:
    set_kaggle_model_visibility(
        owner=owner,
        slug=slug,
        title=title,
        subtitle=subtitle,
        description=description,
        private=False,
    )


def set_huggingface_repo_visibility(
    repo_id: str,
    *,
    repo_type: str,
    private: bool,
) -> None:
    """Set Hugging Face visibility explicitly for staged releases and rollback."""
    try:
        from huggingface_hub import HfApi
    except ImportError as error:
        raise RuntimeError(
            "Install release dependencies with `pip install -e .[release]`."
        ) from error
    HfApi(token=require_huggingface_token()).update_repo_settings(
        repo_id=repo_id,
        repo_type=repo_type,
        private=private,
    )


def publish_huggingface_dataset(
    bundle_dir: str | Path,
    *,
    owner: str,
    slug: str = "falsifyrl-seed",
) -> str:
    try:
        from huggingface_hub import HfApi
    except ImportError as error:
        raise RuntimeError(
            "Install release dependencies with `pip install -e .[release]`."
        ) from error
    repo_id = f"{owner}/{slug}"
    api = HfApi(token=require_huggingface_token())
    api.create_repo(repo_id=repo_id, repo_type="dataset", private=False, exist_ok=True)
    api.upload_folder(
        folder_path=str(bundle_dir),
        repo_id=repo_id,
        repo_type="dataset",
        commit_message="Publish verified FalsifyRL seed dataset v1",
    )
    return f"https://huggingface.co/datasets/{repo_id}"


def publish_kaggle_dataset(
    bundle_dir: str | Path,
    *,
    owner: str,
    slug: str = "falsifyrl-seed",
) -> str:
    try:
        import kagglehub
    except ImportError as error:
        raise RuntimeError(
            "Install release dependencies with `pip install -e .[release]`."
        ) from error
    require_kaggle_token()
    handle = f"{owner}/{slug}"
    kagglehub.dataset_upload(
        handle,
        str(bundle_dir),
        version_notes="Verified FalsifyRL reward-matched seed dataset v1",
    )
    set_kaggle_dataset_public(
        owner=owner,
        slug=slug,
        title=(
            "FalsifyRL AutoScientist-Adapted Dataset"
            if slug == "falsifyrl-adapted"
            else "FalsifyRL Source Reward-Hacking Dataset"
        ),
        subtitle=(
            "Exact audited Adaptive Data export with family-disjoint held-out splits"
            if slug == "falsifyrl-adapted"
            else "Reward-matched embodied multi-agent traces with executable repair labels"
        ),
        description=(
            "Verified FalsifyRL data for evidence-grounded reward-hacking diagnosis "
            "and executable repair in embodied multi-agent reinforcement learning."
        ),
    )
    return f"https://www.kaggle.com/datasets/{handle}"


def audit_model_bundle(bundle_dir: str | Path) -> None:
    bundle = Path(bundle_dir)
    required = {
        "LICENSE",
        "README.md",
        "adapter_config.json",
        "adapter_model.safetensors",
    }
    missing = sorted(filename for filename in required if not (bundle / filename).is_file())
    if missing:
        raise ValueError(f"model bundle is missing required files: {missing}")
    card = (bundle / "README.md").read_text(encoding="utf-8")
    unresolved = sorted(
        marker
        for marker in (
            "BASE_MODEL_ID",
            "DATASET_REPO_ID",
            "AUTOSCIENTIST_RUN_ID",
            "BEST_WIN_RATE",
            "EVALUATION_REPORT_URL",
        )
        if marker in card
    )
    if unresolved:
        raise ValueError(f"model card has unresolved markers: {unresolved}")


def publish_huggingface_model(
    bundle_dir: str | Path,
    *,
    owner: str,
    slug: str = "falsifyrl-autoscientist",
    private: bool = False,
) -> str:
    try:
        from huggingface_hub import HfApi
    except ImportError as error:
        raise RuntimeError(
            "Install release dependencies with `pip install -e .[release]`."
        ) from error
    audit_model_bundle(bundle_dir)
    repo_id = f"{owner}/{slug}"
    api = HfApi(token=require_huggingface_token())
    api.create_repo(repo_id=repo_id, repo_type="model", private=private, exist_ok=True)
    api.update_repo_settings(repo_id=repo_id, repo_type="model", private=private)
    api.upload_folder(
        folder_path=str(bundle_dir),
        repo_id=repo_id,
        repo_type="model",
        commit_message="Publish best FalsifyRL AutoScientist checkpoint",
    )
    return f"https://huggingface.co/{repo_id}"


def publish_kaggle_model(
    bundle_dir: str | Path,
    *,
    owner: str,
    slug: str = "falsifyrl-autoscientist",
    variation: str = "lora",
    license_name: str | None = "Apache 2.0",
    private: bool = False,
) -> str:
    try:
        import kagglehub
    except ImportError as error:
        raise RuntimeError(
            "Install release dependencies with `pip install -e .[release]`."
        ) from error
    require_kaggle_token()
    audit_model_bundle(bundle_dir)
    handle = f"{owner}/{slug}/pytorch/{variation}"
    kagglehub.model_upload(
        handle,
        str(bundle_dir),
        license_name=license_name,
        version_notes="Best FalsifyRL AutoScientist LoRA checkpoint",
    )
    # kagglehub.model_upload does not expose a visibility option. Set the requested
    # visibility immediately after upload; a new model can be briefly visible if
    # Kaggle changes its upload default, so callers should still treat this as a
    # best-effort stage rather than an atomic private transaction.
    set_kaggle_model_visibility(
        owner=owner,
        slug=slug,
        title=slug.replace("-", " "),
        subtitle="LoRA critic for evidence-grounded diagnosis and executable reward repair",
        description=(
            "Best audited AutoScientist checkpoint trained on the exact FalsifyRL "
            "adapted dataset and evaluated on a family-disjoint held-out robotics split."
        ),
        private=private,
    )
    return f"https://www.kaggle.com/models/{handle}"


def publish_huggingface_space(
    bundle_dir: str | Path,
    *,
    owner: str,
    base_model_id: str,
    model_repo_id: str,
    slug: str = "falsifyrl",
    private: bool = False,
) -> str:
    try:
        from huggingface_hub import HfApi
    except ImportError as error:
        raise RuntimeError(
            "Install release dependencies with `pip install -e .[release]`."
        ) from error
    bundle = Path(bundle_dir)
    required = {
        "README.md",
        "app.py",
        "requirements.txt",
        "examples.json",
        "predictions.json",
    }
    missing = sorted(filename for filename in required if not (bundle / filename).is_file())
    if missing:
        raise ValueError(f"Space bundle is missing required files: {missing}")
    examples = json.loads((bundle / "examples.json").read_text(encoding="utf-8"))
    if len(examples) < 16:
        raise ValueError("Space bundle must include eight matched control/exploit pairs")
    prediction_bundle = json.loads(
        (bundle / "predictions.json").read_text(encoding="utf-8")
    )
    if (
        prediction_bundle.get("schema_version") != 1
        or prediction_bundle.get("mode") != "cached_exact_checkpoint_predictions"
        or not isinstance(prediction_bundle.get("source_predictions_sha256"), str)
        or len(prediction_bundle["source_predictions_sha256"]) != 64
    ):
        raise ValueError("Space predictions bundle has an invalid provenance contract")
    predictions = prediction_bundle.get("predictions")
    expected_ids = {str(example["example_id"]) for example in examples}
    if not isinstance(predictions, dict) or set(predictions) != expected_ids:
        raise ValueError("Space predictions must cover exactly the published examples")
    for example_id, prediction in predictions.items():
        try:
            Diagnosis.from_json(json.dumps(prediction, sort_keys=True))
        except (TypeError, ValueError, json.JSONDecodeError) as error:
            raise ValueError(
                f"Space prediction for {example_id} is not a strict diagnosis"
            ) from error

    repo_id = f"{owner}/{slug}"
    token = require_huggingface_token()
    api = HfApi(token=token)
    api.create_repo(
        repo_id=repo_id,
        repo_type="space",
        space_sdk="gradio",
        private=private,
        exist_ok=True,
    )
    api.update_repo_settings(repo_id=repo_id, repo_type="space", private=private)
    api.upload_folder(
        folder_path=bundle,
        repo_id=repo_id,
        repo_type="space",
        commit_message="Publish FalsifyRL interactive critic demo",
    )
    api.add_space_variable(repo_id, "BASE_MODEL_ID", base_model_id)
    api.add_space_variable(repo_id, "MODEL_REPO_ID", model_repo_id)
    return f"https://huggingface.co/spaces/{repo_id}"


def render_model_card(
    template_path: str | Path,
    destination: str | Path,
    *,
    base_model_id: str,
    dataset_repo_id: str,
    autoscientist_run_id: str,
    best_win_rate: float,
    evaluation_report_url: str,
) -> Path:
    content = Path(template_path).read_text(encoding="utf-8")
    replacements = {
        "BASE_MODEL_ID": base_model_id,
        "DATASET_REPO_ID": dataset_repo_id,
        "AUTOSCIENTIST_RUN_ID": autoscientist_run_id,
        "BEST_WIN_RATE": f"{best_win_rate:.4f}",
        "EVALUATION_REPORT_URL": evaluation_report_url,
    }
    for marker, value in replacements.items():
        content = content.replace(marker, value)
    output = Path(destination)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(content, encoding="utf-8")
    return output


def _safe_extract_tar(archive: Path, destination: Path) -> None:
    destination_root = destination.resolve()
    with tarfile.open(archive, mode="r:*") as bundle:
        for member in bundle.getmembers():
            member_path = (destination / member.name).resolve()
            if destination_root not in member_path.parents and member_path != destination_root:
                raise ValueError(f"unsafe checkpoint archive path: {member.name}")
            if member.issym() or member.islnk() or member.isdev():
                raise ValueError(
                    f"checkpoint archive contains unsupported link/device: {member.name}"
                )
        bundle.extractall(destination)


def _decompress_zstandard_archive(archive: Path, destination: Path) -> None:
    try:
        import zstandard
    except ImportError as error:
        raise RuntimeError(
            "Zstandard checkpoint support requires the release dependencies: "
            "`pip install -e .[release]`."
        ) from error

    written = 0
    with (
        archive.open("rb") as source,
        destination.open("wb") as target,
        zstandard.ZstdDecompressor().stream_reader(source) as reader,
    ):
        for chunk in iter(lambda: reader.read(1024 * 1024), b""):
            written += len(chunk)
            if written > MAX_DECOMPRESSED_CHECKPOINT_BYTES:
                raise ValueError(
                    "decompressed checkpoint exceeds the 4 GiB safety limit"
                )
            target.write(chunk)


def _safe_extract_archive(archive: Path, destination: Path) -> None:
    with archive.open("rb") as stream:
        magic = stream.read(len(ZSTANDARD_MAGIC))
    if magic != ZSTANDARD_MAGIC:
        _safe_extract_tar(archive, destination)
        return

    with tempfile.TemporaryDirectory(prefix="falsifyrl-zstandard-") as temporary:
        decompressed = Path(temporary) / "checkpoint.tar"
        _decompress_zstandard_archive(archive, decompressed)
        _safe_extract_tar(decompressed, destination)


def extract_adapter_checkpoint(
    checkpoint_archive: str | Path,
    destination: str | Path,
) -> Path:
    archive = Path(checkpoint_archive)
    extraction_root = Path(destination)
    if extraction_root.exists() and any(extraction_root.iterdir()):
        raise ValueError(
            f"checkpoint extraction destination must be empty: {extraction_root}"
        )
    extraction_root.mkdir(parents=True, exist_ok=True)
    _safe_extract_archive(archive, extraction_root)
    adapter_configs = list(extraction_root.rglob("adapter_config.json"))
    if len(adapter_configs) != 1:
        raise ValueError(
            "checkpoint must contain exactly one adapter_config.json, "
            f"found {len(adapter_configs)}"
        )
    adapter_root = adapter_configs[0].parent
    if not (adapter_root / "adapter_model.safetensors").is_file():
        raise ValueError("checkpoint is missing adapter_model.safetensors")
    return adapter_root


def canonicalize_adapter_base_model(
    adapter_root: str | Path,
    canonical_model_id: str,
) -> str:
    config_path = Path(adapter_root) / "adapter_config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    original_model_id = str(config.get("base_model_name_or_path") or "")
    if not original_model_id:
        raise ValueError("adapter config is missing base_model_name_or_path")

    def model_identity(model_id: str) -> str:
        name = model_id.rsplit("/", 1)[-1].casefold()
        for suffix in ("-reference__tog__ft", "__tog__ft", "-reference"):
            if name.endswith(suffix):
                name = name[: -len(suffix)]
        if name.startswith("meta-"):
            name = name.removeprefix("meta-")
        return "".join(character for character in name if character.isalnum())

    if model_identity(original_model_id) != model_identity(canonical_model_id):
        raise ValueError(
            "checkpoint base model does not match the AutoScientist run model: "
            f"{original_model_id!r} != {canonical_model_id!r}"
        )
    if original_model_id != canonical_model_id:
        config["base_model_name_or_path"] = canonical_model_id
        temporary = config_path.with_suffix(".json.tmp")
        temporary.write_text(
            json.dumps(config, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temporary.replace(config_path)
    return original_model_id


def prepare_model_bundle(
    checkpoint_archive: str | Path,
    bundle_dir: str | Path,
    *,
    base_model_id: str,
    dataset_repo_id: str,
    autoscientist_run_id: str,
    best_win_rate: float,
    evaluation_report: str | Path,
    base_predictions: str | Path | None = None,
    adapted_predictions: str | Path | None = None,
    evaluation_metadata_dir: str | Path | None = None,
    model_card_template: str | Path = "release/model/README.md",
    license_path: str | Path = "release/model/LICENSE",
) -> dict[str, Any]:
    archive = Path(checkpoint_archive)
    destination = Path(bundle_dir)
    if destination.exists() and any(destination.iterdir()):
        raise ValueError(f"model bundle destination must be empty: {destination}")
    destination.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="falsifyrl-checkpoint-") as temporary:
        adapter_root = extract_adapter_checkpoint(archive, temporary)
        for source in adapter_root.iterdir():
            target = destination / source.name
            if source.is_dir():
                shutil.copytree(source, target)
            else:
                shutil.copy2(source, target)
    checkpoint_base_model_id = canonicalize_adapter_base_model(
        destination,
        base_model_id,
    )

    report_source = Path(evaluation_report)
    if not report_source.is_file():
        raise FileNotFoundError(report_source)
    report_target = destination / "evaluation-report.json"
    shutil.copy2(report_source, report_target)
    prediction_sources = {
        "falsifyrl-base-test-predictions.jsonl": base_predictions,
        "falsifyrl-adapted-test-predictions.jsonl": adapted_predictions,
    }
    provided_predictions = [path is not None for path in prediction_sources.values()]
    if any(provided_predictions) and not all(provided_predictions):
        raise ValueError("base and adapted prediction evidence must be provided together")
    for filename, source in prediction_sources.items():
        if source is None:
            continue
        source_path = Path(source)
        if not source_path.is_file():
            raise FileNotFoundError(source_path)
        shutil.copy2(source_path, destination / filename)
    if evaluation_metadata_dir is not None:
        metadata_root = Path(evaluation_metadata_dir)
        for filename in ("evaluation-manifest.json", "colab-evaluation.json"):
            source_path = metadata_root / filename
            if not source_path.is_file():
                raise FileNotFoundError(source_path)
            shutil.copy2(source_path, destination / filename)
    shutil.copy2(license_path, destination / "LICENSE")
    render_model_card(
        model_card_template,
        destination / "README.md",
        base_model_id=base_model_id,
        dataset_repo_id=dataset_repo_id,
        autoscientist_run_id=autoscientist_run_id,
        best_win_rate=best_win_rate,
        evaluation_report_url=report_target.name,
    )
    audit_model_bundle(destination)

    files = {
        path.relative_to(destination).as_posix(): {
            "bytes": path.stat().st_size,
            "sha256": _sha256(path),
        }
        for path in sorted(destination.rglob("*"))
        if path.is_file()
    }
    manifest = {
        "artifact": "falsifyrl-autoscientist",
        "base_model_id": base_model_id,
        "checkpoint_original_base_model_id": checkpoint_base_model_id,
        "dataset_repo_id": dataset_repo_id,
        "autoscientist_run_id": autoscientist_run_id,
        "best_win_rate": best_win_rate,
        "files": files,
    }
    (destination / "release-manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest
