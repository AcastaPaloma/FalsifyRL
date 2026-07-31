from __future__ import annotations

import csv
import hashlib
import io
import json
import os
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
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
    training_dataset_id: str | None = None
    training_dataset_status: str | None = None
    training_prompt_column: str | None = None
    training_completion_column: str | None = None
    training_dataset_export_sha256: str | None = None
    autoscientist_run_id: str | None = None
    autoscientist_run_ids: list[str] = field(default_factory=list)
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
            "training_dataset_id": self.training_dataset_id,
            "training_dataset_status": self.training_dataset_status,
            "training_prompt_column": self.training_prompt_column,
            "training_completion_column": self.training_completion_column,
            "training_dataset_export_sha256": self.training_dataset_export_sha256,
            "autoscientist_run_id": self.autoscientist_run_id,
            "autoscientist_run_ids": self.autoscientist_run_ids,
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


def _select_adapted_columns(
    fieldnames: list[str],
    rows: list[dict[str, str]],
) -> tuple[str, str]:
    fields = set(fieldnames)
    prompt_column = next(
        (
            name
            for name in ("enhanced_prompt", "prompt")
            if name in fields and all((row.get(name) or "").strip() for row in rows)
        ),
        None,
    )
    identity_column = next(
        (
            name
            for name in ("original_prompt", "prompt", prompt_column)
            if name in fields and all((row.get(name) or "").strip() for row in rows)
        ),
        None,
    )
    if not prompt_column or not identity_column:
        raise ValueError(
            "adapted CSV must expose a fully populated prompt and source identity column"
        )
    return prompt_column, identity_column


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
        source_completions: dict[str, str] = {}
        source_row_count = 0
        for row in source_reader:
            source_row_count += 1
            prompt = row["prompt"]
            if prompt in source_rows:
                if row["completion"] != source_completions[prompt]:
                    raise ValueError(
                        "source training CSV contains a prompt with conflicting completions"
                    )
                continue
            source_rows[prompt] = Diagnosis.from_json(row["completion"])
            source_completions[prompt] = row["completion"]

    if source_row_count != expected_rows:
        raise ValueError(
            f"source training row count {source_row_count} != expected {expected_rows}"
        )
    source_unique_row_count = len(source_rows)

    with adapted.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if not reader.fieldnames or len(reader.fieldnames) != len(set(reader.fieldnames)):
            raise ValueError("adapted CSV has missing or duplicate headers")
        adapted_rows = list(reader)
        prompt_column, identity_column = _select_adapted_columns(
            reader.fieldnames,
            adapted_rows,
        )
        seen_prompts: set[str] = set()
        for row_number, row in enumerate(adapted_rows, start=2):
            training_prompt = row.get(prompt_column, "")
            identity_prompt = row.get(identity_column, "")
            if not training_prompt.strip():
                raise ValueError(f"adapted row {row_number} has a blank training prompt")
            if identity_prompt not in source_rows:
                raise ValueError(
                    f"adapted row {row_number} cannot be matched to a source prompt"
                )
            if identity_prompt in seen_prompts:
                raise ValueError(f"adapted row {row_number} duplicates a source prompt")
            seen_prompts.add(identity_prompt)

        completion_candidates = [
            name
            for name in ("enhanced_completion", "completion")
            if name in set(reader.fieldnames)
            and all((row.get(name) or "").strip() for row in adapted_rows)
        ]
        completion_column = None
        rejected_completion_columns: list[str] = []
        for candidate in completion_candidates:
            candidate_valid = True
            for row in adapted_rows:
                try:
                    diagnosis = Diagnosis.from_json(row[candidate])
                except (KeyError, TypeError, ValueError, json.JSONDecodeError):
                    candidate_valid = False
                    break
                if _diagnosis_invariants(diagnosis) != _diagnosis_invariants(
                    source_rows[row[identity_column]]
                ):
                    candidate_valid = False
                    break
            if candidate_valid:
                completion_column = candidate
                break
            rejected_completion_columns.append(candidate)
        if completion_column is None:
            raise ValueError(
                "adapted CSV has no fully valid label-preserving completion column"
            )

    if len(seen_prompts) != source_unique_row_count:
        raise ValueError(
            "adapted training row count "
            f"{len(seen_prompts)} != exact-deduplicated source count "
            f"{source_unique_row_count}"
        )
    if seen_prompts != set(source_rows):
        raise ValueError("adapted CSV does not contain every source training prompt exactly once")
    return {
        "schema_version": "1.1",
        "dataset_variant": "adapted",
        "row_count": len(seen_prompts),
        "source_row_count": source_row_count,
        "source_unique_row_count": source_unique_row_count,
        "exact_duplicate_rows_collapsed": (
            source_row_count - source_unique_row_count
        ),
        "prompt_column": prompt_column,
        "completion_column": completion_column,
        "identity_column": identity_column,
        "rejected_completion_columns": rejected_completion_columns,
        "enhanced_prompt_selected": prompt_column == "enhanced_prompt",
        "enhanced_completion_selected": completion_column == "enhanced_completion",
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

    download_result = client.datasets.download(state.dataset_id, file_format="csv")
    if not isinstance(download_result, str):
        raise TypeError("dataset download did not return text")
    parsed = urlparse(download_result)
    if parsed.scheme or parsed.netloc:
        if parsed.scheme != "https" or not parsed.netloc:
            raise ValueError("dataset download returned a non-HTTPS URL")
        content = (
            _fetch_https(download_result)
            if fetcher is None
            else fetcher(download_result)
        )
    else:
        if "\n" not in download_result and "\r" not in download_result:
            raise ValueError(
                "dataset download returned neither CSV text nor an absolute HTTPS URL"
            )
        content = download_result.encode("utf-8")
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


def prepare_training_dataset(
    client: Any,
    state: WorkflowState,
    *,
    timeout: float = 3600,
    on_dataset_created: Callable[[WorkflowState], None] | None = None,
) -> WorkflowState:
    if (
        not state.adapted_schema_valid
        or not state.adapted_export_path
        or not state.adapted_export_sha256
        or not state.adapted_row_count
        or not state.adapted_prompt_column
        or not state.adapted_completion_column
    ):
        raise ValueError("an audited adapted export is required before passthrough upload")
    export_path = Path(state.adapted_export_path)
    if not export_path.is_file() or _sha256(export_path) != state.adapted_export_sha256:
        raise ValueError("adapted export file does not match the audited SHA-256")

    if not state.training_dataset_id:
        created = client.datasets.upload_file(
            export_path,
            name="falsifyrl-autoscientist-train-v1",
            processing_mode="passthrough",
            column_mapping={
                "prompt": state.adapted_prompt_column,
                "completion": state.adapted_completion_column,
            },
        )
        state.training_dataset_id = str(_value(created, "dataset_id"))
        state.training_dataset_status = str(_value(created, "status"))
        if on_dataset_created is not None:
            on_dataset_created(state)

    if state.training_dataset_status != "succeeded":
        completed = client.datasets.wait_for_completion(
            state.training_dataset_id,
            timeout=timeout,
        )
        state.training_dataset_status = str(_value(completed, "status"))
    if state.training_dataset_status != "succeeded":
        raise RuntimeError(
            f"training dataset ended with status {state.training_dataset_status}"
        )

    dataset = client.datasets.get(state.training_dataset_id)
    if int(_value(dataset, "row_count", 0)) != state.adapted_row_count:
        raise ValueError("passthrough training dataset row count does not match export")
    mapping = _value(dataset, "configured_column_mapping", {}) or {}
    if mapping and (
        _value(mapping, "prompt") != state.adapted_prompt_column
        or _value(mapping, "completion") != state.adapted_completion_column
    ):
        raise ValueError("passthrough training dataset mapping does not match audit")

    downloaded = client.datasets.download(state.training_dataset_id, file_format="csv")
    if not isinstance(downloaded, str):
        raise TypeError("passthrough training dataset download did not return text")
    with export_path.open("r", encoding="utf-8-sig", newline="") as stream:
        source_rows = list(csv.DictReader(stream))
    training_rows = list(csv.DictReader(io.StringIO(downloaded, newline="")))
    if not training_rows or set(training_rows[0]) != {
        "original_prompt",
        "original_completion",
        "enhanced_prompt",
        "enhanced_completion",
    }:
        raise ValueError("passthrough training dataset has an unexpected canonical schema")
    source_pairs = [
        (
            row[state.adapted_prompt_column],
            row[state.adapted_completion_column],
        )
        for row in source_rows
    ]
    training_pairs = [
        (row["original_prompt"], row["original_completion"])
        for row in training_rows
    ]
    if training_pairs != source_pairs:
        raise ValueError("passthrough training dataset content does not match audit")
    state.training_prompt_column = "original_prompt"
    state.training_completion_column = "original_completion"
    state.training_dataset_export_sha256 = hashlib.sha256(
        downloaded.encode("utf-8")
    ).hexdigest()
    return state


def ingest_and_estimate(
    client: Any,
    state: WorkflowState,
    *,
    ingestion_timeout: float = 3600,
    on_dataset_created: Callable[[WorkflowState], None] | None = None,
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
        if on_dataset_created is not None:
            on_dataset_created(state)

    deadline = time.monotonic() + ingestion_timeout
    while True:
        ingested = client.datasets.get(state.dataset_id)
        status = str(_value(ingested, "status"))
        if status in {"awaiting_input", "ready", "succeeded"}:
            break
        if status == "failed":
            raise RuntimeError("dataset ingestion failed")
        if time.monotonic() >= deadline:
            raise TimeoutError(f"dataset ingestion timed out with status {status}")
        time.sleep(2)

    state.dataset_status = str(_value(ingested, "status"))
    if state.dataset_status not in {"awaiting_input", "succeeded", "ready"}:
        raise RuntimeError(f"dataset ingestion ended with status {state.dataset_status}")
    row_count = _value(ingested, "row_count")
    if row_count is None:
        raise RuntimeError("ingested dataset does not report a row count")
    if int(row_count) != plan.expected_training_rows:
        raise ValueError(
            f"ingested row count {row_count} != expected {plan.expected_training_rows}"
        )

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
    on_adaptation_started: Callable[[WorkflowState], None] | None = None,
) -> WorkflowState:
    if not state.dataset_id:
        raise ValueError("ingest and estimate the dataset before adaptation")
    run = client.datasets.run(
        state.dataset_id,
        column_mapping={"prompt": "prompt", "completion": "completion"},
        training_type="instruction_dataset",
        job_specification={
            "idempotency_key": f"falsifyrl-adapt-v1-{state.dataset_id}",
        },
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
    if on_adaptation_started is not None:
        on_adaptation_started(state)
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
    on_run_started: Callable[[WorkflowState], None] | None = None,
) -> WorkflowState:
    if (
        not state.dataset_id
        or state.dataset_status != "succeeded"
        or not state.adapted_schema_valid
        or not state.adapted_row_count
        or state.adapted_row_count > state.plan.expected_training_rows
        or not state.adapted_export_sha256
        or not state.adapted_audit_sha256
        or not state.adapted_prompt_column
        or not state.adapted_completion_column
        or not state.training_dataset_id
        or state.training_dataset_status != "succeeded"
        or not state.training_prompt_column
        or not state.training_completion_column
    ):
        raise ValueError(
            "an exported and audited exact adapted dataset is required before training"
        )
    arguments: dict[str, Any] = {
        "dataset_id": state.training_dataset_id,
        "max_iterations": state.plan.max_iterations,
        "target_win_rate": state.plan.target_win_rate,
        "data_format": "instruction",
        "column_mapping": {
            "prompt": state.training_prompt_column,
            "completion": state.training_completion_column,
        },
        "idempotency_key": (
            f"falsifyrl-v2-{state.training_dataset_id}"
        ),
    }
    if state.plan.model:
        arguments["model"] = state.plan.model

    created = client.autoscientist.create(**arguments)
    if (
        state.autoscientist_run_id
        and state.autoscientist_run_id not in state.autoscientist_run_ids
    ):
        state.autoscientist_run_ids.append(state.autoscientist_run_id)
    state.autoscientist_run_id = str(_value(created, "id"))
    if state.autoscientist_run_id not in state.autoscientist_run_ids:
        state.autoscientist_run_ids.append(state.autoscientist_run_id)
    created_model = _value(created, "model")
    if created_model is not None:
        state.resolved_model = str(created_model)
    if on_run_started is not None:
        on_run_started(state)
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
