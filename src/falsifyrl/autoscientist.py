from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

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
    return state


def run_autoscientist(
    client: Any,
    state: WorkflowState,
    *,
    timeout: float = 14_400,
) -> WorkflowState:
    if not state.dataset_id or state.dataset_status != "succeeded":
        raise ValueError("a succeeded adapted dataset is required before training")
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
