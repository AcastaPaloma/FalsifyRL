from __future__ import annotations

import csv
import hashlib
import json
import os
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlparse
from urllib.request import Request, urlopen

from falsifyrl.schema import Diagnosis

SUPPORTED_SOURCES = frozenset({"file", "huggingface", "kaggle"})


@dataclass(frozen=True)
class AutoScientistPlan:
    source: str
    source_url: str | None = None
    source_file: str = "train.csv"
    local_file: str | None = None
    max_iterations: int = 3
    target_win_rate: float = 0.75
    model: str | None = None
    expected_training_rows: int = 2560

    def __post_init__(self) -> None:
        if self.source not in SUPPORTED_SOURCES:
            raise ValueError(f"unsupported dataset source: {self.source}")
        if self.source == "file" and not self.local_file:
            raise ValueError("file source requires local_file")
        if self.source != "file" and not self.source_url:
            raise ValueError(f"{self.source} source requires source_url")
        if self.max_iterations < 1 or self.max_iterations > 10:
            raise ValueError("max_iterations must be in [1, 10]")
        if self.target_win_rate <= 0.0 or self.target_win_rate > 1.0:
            raise ValueError("target_win_rate must be in (0, 1]")
        if self.expected_training_rows < 1:
            raise ValueError("expected_training_rows must be positive")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class WorkflowState:
    plan: AutoScientistPlan
    dataset_id: str | None = None
    dataset_status: str | None = None
    estimated_credits: float | None = None
    estimated_minutes: float | None = None
    adaptation_run_id: str | None = None
    adapted_export_path: str | None = None
    adapted_export_sha256: str | None = None
    adapted_audit_path: str | None = None
    adapted_audit_sha256: str | None = None
    adapted_row_count: int | None = None
    adapted_prompt_column: str | None = None
    adapted_completion_column: str | None = None
    adapted_schema_valid: bool = False
    autoscientist_run_id: str | None = None
    autoscientist_status: str | None = None
    best_win_rate: float | None = None
    resolved_model: str | None = None
    download_available: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "plan": self.plan.to_dict(),
            "dataset_id": self.dataset_id,
            "dataset_status": self.dataset_status,
            "estimated_credits": self.estimated_credits,
            "estimated_minutes": self.estimated_minutes,
            "adaptation_run_id": self.adaptation_run_id,
            "adapted_export_path": self.adapted_export_path,
            "adapted_export_sha256": self.adapted_export_sha256,
            "adapted_audit_path": self.adapted_audit_path,
            "adapted_audit_sha256": self.adapted_audit_sha256,
            "adapted_row_count": self.adapted_row_count,
            "adapted_prompt_column": self.adapted_prompt_column,
            "adapted_completion_column": self.adapted_completion_column,
            "adapted_schema_valid": self.adapted_schema_valid,
            "autoscientist_run_id": self.autoscientist_run_id,
            "autoscientist_status": self.autoscientist_status,
            "best_win_rate": self.best_win_rate,
            "resolved_model": self.resolved_model,
            "download_available": self.download_available,
        }

    def save(self, path: str | Path) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: str | Path) -> WorkflowState:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(plan=AutoScientistPlan(**data.pop("plan")), **data)


def require_api_key() -> str:
    api_key = os.environ.get("ADAPTION_API_KEY")
    if not api_key:
        raise RuntimeError(
            "ADAPTION_API_KEY is required for external workflow actions. "
            "Set it in the process environment; never pass it as a command-line argument."
        )
    return api_key


def create_client() -> Any:
    try:
        from adaption import Adaption
    except ImportError as error:
        raise RuntimeError(
            "Install the platform integration with `pip install -e .[platforms]`."
        ) from error
    return Adaption(api_key=require_api_key())


def _value(resource: Any, name: str, default: Any = None) -> Any:
    if isinstance(resource, dict):
        return resource.get(name, default)
    return getattr(resource, name, default)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _fetch_https(url: str, *, max_bytes: int = 100 * 1024 * 1024) -> bytes:
    parsed = urlparse(url)
    if parsed.scheme != "https" or not parsed.netloc:
        raise ValueError("adapted dataset download must use an HTTPS URL")
    request = Request(url, headers={"User-Agent": "FalsifyRL/0.1"})
    with urlopen(request, timeout=300) as response:  # noqa: S310
        content = response.read(max_bytes + 1)
    if len(content) > max_bytes:
        raise ValueError("adapted dataset export exceeds the 100 MiB safety limit")
    return content


def _select_adapted_columns(fieldnames: list[str]) -> tuple[str, str, str]:
    fields = set(fieldnames)
    prompt_column = next(
        (name for name in ("enhanced_prompt", "prompt") if name in fields),
        None,
    )
    completion_column = next(
        (name for name in ("enhanced_completion", "completion") if name in fields),
        None,
    )
    identity_column = next(
        (name for name in ("original_prompt", "prompt", prompt_column) if name in fields),
        None,
    )
    if not prompt_column or not completion_column or not identity_column:
        raise ValueError(
            "adapted CSV must expose enhanced_prompt/enhanced_completion or prompt/completion"
        )
    return prompt_column, completion_column, identity_column


def _diagnosis_invariants(diagnosis: Diagnosis) -> dict[str, Any]:
    values = diagnosis.to_dict()
    return {
        key: values[key]
        for key in (
            "verdict",
            "failure_type",
            "responsible_agents",
            "evidence_steps",
            "counterexample_config",
            "reward_patch",
        )
    }


def audit_adapted_csv(
    adapted_path: str | Path,
    source_path: str | Path,
    *,
    expected_rows: int,
) -> dict[str, Any]:
    adapted = Path(adapted_path)
    source = Path(source_path)
    with source.open("r", encoding="utf-8", newline="") as stream:
        source_reader = csv.DictReader(stream)
        if source_reader.fieldnames != ["prompt", "completion"]:
            raise ValueError("source training CSV must have exactly prompt,completion columns")
        source_rows: dict[str, Diagnosis] = {}
        for row in source_reader:
            prompt = row["prompt"]
            if prompt in source_rows:
                raise ValueError("source training CSV contains duplicate prompts")
            source_rows[prompt] = Diagnosis.from_json(row["completion"])

    if len(source_rows) != expected_rows:
        raise ValueError(
            f"source training row count {len(source_rows)} != expected {expected_rows}"
        )

    with adapted.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if not reader.fieldnames or len(reader.fieldnames) != len(set(reader.fieldnames)):
            raise ValueError("adapted CSV has missing or duplicate headers")
        prompt_column, completion_column, identity_column = _select_adapted_columns(
            reader.fieldnames
        )
        seen_prompts: set[str] = set()
        for row_number, row in enumerate(reader, start=2):
            training_prompt = row.get(prompt_column, "")
            identity_prompt = row.get(identity_column, "")
            completion = row.get(completion_column, "")
            if not training_prompt.strip():
                raise ValueError(f"adapted row {row_number} has a blank training prompt")
            if identity_prompt not in source_rows:
                raise ValueError(
                    f"adapted row {row_number} cannot be matched to a source prompt"
                )
            if identity_prompt in seen_prompts:
                raise ValueError(f"adapted row {row_number} duplicates a source prompt")
            try:
                diagnosis = Diagnosis.from_json(completion)
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
                raise ValueError(
                    f"adapted row {row_number} has an invalid strict completion"
                ) from error
            if _diagnosis_invariants(diagnosis) != _diagnosis_invariants(
                source_rows[identity_prompt]
            ):
                raise ValueError(
                    f"adapted row {row_number} changed executable diagnosis invariants"
                )
            seen_prompts.add(identity_prompt)

    if len(seen_prompts) != expected_rows:
        raise ValueError(
            f"adapted training row count {len(seen_prompts)} != expected {expected_rows}"
        )
    if seen_prompts != set(source_rows):
        raise ValueError("adapted CSV does not contain every source training prompt exactly once")
    return {
        "schema_version": "1.0",
        "dataset_variant": "adapted",
        "row_count": len(seen_prompts),
        "prompt_column": prompt_column,
        "completion_column": completion_column,
        "identity_column": identity_column,
        "all_source_prompts_matched": True,
        "all_completions_strict_json": True,
        "all_diagnosis_invariants_preserved": True,
        "source_sha256": _sha256(source),
        "adapted_sha256": _sha256(adapted),
    }


def export_and_audit_adapted_dataset(
    client: Any,
    state: WorkflowState,
    destination: str | Path,
    source_training_csv: str | Path,
    *,
    fetcher: Callable[[str], bytes] | None = None,
) -> WorkflowState:
    if (
        not state.dataset_id
        or state.dataset_status != "succeeded"
        or not state.adaptation_run_id
    ):
        raise ValueError("a succeeded adaptation run is required before export")

    download_url = client.datasets.download(state.dataset_id, file_format="csv")
    if not isinstance(download_url, str):
        raise TypeError("dataset download did not return a URL")
    parsed = urlparse(download_url)
    if parsed.scheme != "https" or not parsed.netloc:
        raise ValueError("dataset download returned a non-HTTPS URL")
    content = _fetch_https(download_url) if fetcher is None else fetcher(download_url)
    if not isinstance(content, bytes):
        raise TypeError("adapted dataset fetcher must return bytes")

    output_path = Path(destination)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")
    temporary_path.write_bytes(content)
    temporary_path.replace(output_path)

    audit = audit_adapted_csv(
        output_path,
        source_training_csv,
        expected_rows=state.plan.expected_training_rows,
    )
    audit.update(
        {
            "dataset_id": state.dataset_id,
            "adaptation_run_id": state.adaptation_run_id,
            "export_file": output_path.name,
            "source_file": Path(source_training_csv).name,
        }
    )
    audit_path = output_path.with_name(f"{output_path.stem}.audit.json")
    audit_path.write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    state.adapted_export_path = str(output_path.resolve())
    state.adapted_export_sha256 = str(audit["adapted_sha256"])
    state.adapted_audit_path = str(audit_path.resolve())
    state.adapted_audit_sha256 = _sha256(audit_path)
    state.adapted_row_count = int(audit["row_count"])
    state.adapted_prompt_column = str(audit["prompt_column"])
    state.adapted_completion_column = str(audit["completion_column"])
    state.adapted_schema_valid = True
    return state


def ingest_and_estimate(
    client: Any,
    state: WorkflowState,
    *,
    ingestion_timeout: float = 3600,
) -> WorkflowState:
    plan = state.plan
    if state.dataset_id is None:
        if plan.source == "huggingface":
            created = client.datasets.create_from_huggingface(
                url=plan.source_url,
                files=[plan.source_file],
            )
        elif plan.source == "kaggle":
            created = client.datasets.create_from_kaggle(
                url=plan.source_url,
                files=[plan.source_file],
            )
        else:
            created = client.datasets.upload_file(
                plan.local_file,
                name="falsifyrl-seed-v1",
            )
        state.dataset_id = str(_value(created, "dataset_id"))

    ingested = client.datasets.wait_for_completion(
        state.dataset_id,
        timeout=ingestion_timeout,
    )
    state.dataset_status = str(_value(ingested, "status"))
    if state.dataset_status not in {"succeeded", "ready"}:
        raise RuntimeError(f"dataset ingestion ended with status {state.dataset_status}")

    estimate = client.datasets.run(
        state.dataset_id,
        column_mapping={"prompt": "prompt", "completion": "completion"},
        training_type="instruction_dataset",
        recipe_specification={
            "version": "1",
            "recipes": {
                "deduplication": False,
                "prompt_rephrase": False,
                "reasoning_traces": False,
            },
        },
        brand_controls={
            "length": "minimal",
            "blueprint": (
                "Preserve the supplied evidence and exact compact JSON output schema. "
                "Do not add Markdown or hidden reasoning."
            ),
        },
        estimate=True,
    )
    state.estimated_credits = float(_value(estimate, "estimated_credits_consumed", 0))
    state.estimated_minutes = float(_value(estimate, "estimated_minutes", 0))
    return state


def run_adaptation(
    client: Any,
    state: WorkflowState,
    *,
    timeout: float = 7200,
) -> WorkflowState:
    if not state.dataset_id:
        raise ValueError("ingest and estimate the dataset before adaptation")
    run = client.datasets.run(
        state.dataset_id,
        column_mapping={"prompt": "prompt", "completion": "completion"},
        training_type="instruction_dataset",
        recipe_specification={
            "version": "1",
            "recipes": {
                "deduplication": False,
                "prompt_rephrase": False,
                "reasoning_traces": False,
            },
        },
        brand_controls={
            "length": "minimal",
            "blueprint": (
                "Preserve the supplied evidence and exact compact JSON output schema. "
                "Do not add Markdown or hidden reasoning."
            ),
        },
    )
    state.adaptation_run_id = str(_value(run, "run_id"))
    completed = client.datasets.wait_for_completion(state.dataset_id, timeout=timeout)
    state.dataset_status = str(_value(completed, "status"))
    if state.dataset_status != "succeeded":
        raise RuntimeError(f"dataset adaptation ended with status {state.dataset_status}")
    state.adapted_schema_valid = False
    return state


def run_autoscientist(
    client: Any,
    state: WorkflowState,
    *,
    timeout: float = 14_400,
) -> WorkflowState:
    if (
        not state.dataset_id
        or state.dataset_status != "succeeded"
        or not state.adapted_schema_valid
        or state.adapted_row_count != state.plan.expected_training_rows
        or not state.adapted_export_sha256
        or not state.adapted_audit_sha256
    ):
        raise ValueError(
            "an exported and audited exact adapted dataset is required before training"
        )
    arguments: dict[str, Any] = {
        "dataset_id": state.dataset_id,
        "max_iterations": state.plan.max_iterations,
        "target_win_rate": state.plan.target_win_rate,
        "data_format": "instruction",
        "idempotency_key": f"falsifyrl-v1-{state.dataset_id}",
    }
    if state.plan.model:
        arguments["model"] = state.plan.model

    created = client.autoscientist.create(**arguments)
    state.autoscientist_run_id = str(_value(created, "id"))
    completed = client.autoscientist.wait_for_completion(
        state.autoscientist_run_id,
        timeout=timeout,
    )
    state.autoscientist_status = str(_value(completed, "status"))
    best_win_rate = _value(completed, "best_win_rate")
    state.best_win_rate = None if best_win_rate is None else float(best_win_rate)
    resolved_model = _value(completed, "model")
    state.resolved_model = None if resolved_model is None else str(resolved_model)
    state.download_available = bool(_value(completed, "download_available", False))
    if state.autoscientist_status != "succeeded":
        raise RuntimeError(
            f"AutoScientist ended with status {state.autoscientist_status}"
        )
    return state


def refresh_status(client: Any, state: WorkflowState) -> WorkflowState:
    if not state.autoscientist_run_id:
        raise ValueError("state has no AutoScientist run ID")
    run = client.autoscientist.get(state.autoscientist_run_id)
    state.autoscientist_status = str(_value(run, "status"))
    best_win_rate = _value(run, "best_win_rate")
    state.best_win_rate = None if best_win_rate is None else float(best_win_rate)
    state.download_available = bool(_value(run, "download_available", False))
    return state


def download_checkpoint(
    client: Any,
    state: WorkflowState,
    destination: str | Path,
) -> Path:
    if not state.autoscientist_run_id or not state.download_available:
        raise ValueError("the best checkpoint is not available for download")
    output_path = Path(destination)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with client.autoscientist.with_streaming_response.download(
        state.autoscientist_run_id
    ) as response:
        response.stream_to_file(output_path)
    return output_path
