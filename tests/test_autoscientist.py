from __future__ import annotations

import csv
import hashlib
import io
import json
from types import SimpleNamespace

import pytest

from falsifyrl.autoscientist import (
    AutoScientistPlan,
    WorkflowState,
    export_and_audit_adapted_dataset,
    ingest_and_estimate,
    prepare_training_dataset,
    require_api_key,
    run_adaptation,
    run_autoscientist,
)
from scripts.continue_autoscientist import await_existing_adaptation


class FakeDatasets:
    def __init__(self) -> None:
        self.run_calls: list[dict] = []
        self.upload_calls: list[dict] = []

    def create_from_huggingface(self, **kwargs):
        assert kwargs["files"] == ["train.csv"]
        return SimpleNamespace(dataset_id="dataset-123")

    def get(self, dataset_id):
        if dataset_id == "training-456":
            mapping = (
                self.upload_calls[-1]["column_mapping"]
                if self.upload_calls
                else {
                    "prompt": "enhanced_prompt",
                    "completion": "enhanced_completion",
                }
            )
            return SimpleNamespace(
                status="succeeded",
                row_count=2,
                configured_column_mapping=mapping,
            )
        assert dataset_id == "dataset-123"
        return SimpleNamespace(status="awaiting_input", row_count=1)

    def upload_file(self, path, **kwargs):
        self.upload_calls.append({"path": path, **kwargs})
        return SimpleNamespace(dataset_id="training-456", status="succeeded")

    def wait_for_completion(self, dataset_id, timeout):
        assert dataset_id == "dataset-123"
        assert timeout > 0
        return SimpleNamespace(status="succeeded")

    def run(self, dataset_id, **kwargs):
        assert dataset_id == "dataset-123"
        self.run_calls.append(kwargs)
        if not kwargs.get("estimate", False):
            return SimpleNamespace(
                run_id="adaptation-789",
                estimated_credits_consumed=42,
                estimated_minutes=7,
            )
        return SimpleNamespace(
            estimated_credits_consumed=42,
            estimated_minutes=7,
        )

    def download(self, dataset_id, file_format):
        if dataset_id == "training-456":
            return (
                "original_prompt,original_completion,"
                "enhanced_prompt,enhanced_completion\n"
                "p2,c2,,\n"
                "p1,c1,,\n"
            )
        assert dataset_id == "dataset-123"
        assert file_format == "csv"
        return "https://example.test/adapted.csv"


class FakeAutoScientist:
    def __init__(self) -> None:
        self.create_arguments = None

    def create(self, **kwargs):
        self.create_arguments = kwargs
        return SimpleNamespace(id="experiment-456")

    def wait_for_completion(self, experiment_id, timeout):
        assert experiment_id == "experiment-456"
        assert timeout > 0
        return SimpleNamespace(
            status="succeeded",
            best_win_rate=0.81,
            model="small-model",
            download_available=True,
        )


class FakeClient:
    def __init__(self) -> None:
        self.datasets = FakeDatasets()
        self.autoscientist = FakeAutoScientist()


def _plan() -> AutoScientistPlan:
    return AutoScientistPlan(
        source="huggingface",
        source_url="https://huggingface.co/datasets/example/falsifyrl",
        expected_training_rows=1,
    )


def test_plan_rejects_missing_source_location() -> None:
    with pytest.raises(ValueError, match="requires source_url"):
        AutoScientistPlan(source="huggingface")


def test_api_key_is_environment_only(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ADAPTION_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="process environment"):
        require_api_key()

    monkeypatch.setenv("ADAPTION_API_KEY", "test-secret")
    assert require_api_key() == "test-secret"


def test_ingestion_estimate_preserves_exact_column_mapping() -> None:
    client = FakeClient()
    state = ingest_and_estimate(client, WorkflowState(plan=_plan()))

    assert state.dataset_id == "dataset-123"
    assert state.estimated_credits == 42
    assert state.estimated_minutes == 7
    estimate_call = client.datasets.run_calls[0]
    assert estimate_call["column_mapping"] == {
        "prompt": "prompt",
        "completion": "completion",
    }
    assert estimate_call["recipe_specification"]["recipes"] == {
        "deduplication": False,
        "prompt_rephrase": False,
        "reasoning_traces": False,
    }
    assert estimate_call["estimate"] is True


def test_training_uses_idempotency_and_records_submission_ids() -> None:
    client = FakeClient()
    state = WorkflowState(
        plan=_plan(),
        dataset_id="dataset-123",
        dataset_status="succeeded",
        adapted_export_sha256="a" * 64,
        adapted_audit_sha256="b" * 64,
        adapted_row_count=1,
        adapted_prompt_column="prompt",
        adapted_completion_column="completion",
        adapted_schema_valid=True,
        training_dataset_id="training-456",
        training_dataset_status="succeeded",
        training_prompt_column="original_prompt",
        training_completion_column="original_completion",
    )

    snapshots = []
    result = run_autoscientist(
        client,
        state,
        on_run_started=lambda current: snapshots.append(current.to_dict()),
    )

    assert result.autoscientist_run_id == "experiment-456"
    assert result.best_win_rate == 0.81
    assert result.download_available is True
    assert snapshots[0]["autoscientist_run_id"] == "experiment-456"
    arguments = client.autoscientist.create_arguments
    assert arguments["data_format"] == "instruction"
    assert arguments["column_mapping"] == {
        "prompt": "original_prompt",
        "completion": "original_completion",
    }
    assert arguments["dataset_id"] == "training-456"
    assert arguments["idempotency_key"] == "falsifyrl-v2-training-456"
    assert "training_type" not in arguments
    assert "api_key" not in result.to_dict()


def test_training_refinement_uses_distinct_hyperparameter_identity() -> None:
    client = FakeClient()
    state = WorkflowState(
        plan=AutoScientistPlan(
            source="file",
            local_file="train.jsonl",
            expected_training_rows=1,
            model="meta-llama/Llama-3.2-3B-Instruct",
            hyperparams={"learning_rate": 2e-4, "n_epochs": 3.0},
        ),
        dataset_id="dataset-123",
        dataset_status="succeeded",
        adapted_export_sha256="a" * 64,
        adapted_audit_sha256="b" * 64,
        adapted_row_count=1,
        adapted_prompt_column="prompt",
        adapted_completion_column="completion",
        adapted_schema_valid=True,
        training_dataset_id="training-456",
        training_dataset_status="succeeded",
        training_prompt_column="original_prompt",
        training_completion_column="original_completion",
        autoscientist_run_id="prior-run",
        autoscientist_run_ids=["prior-run"],
        autoscientist_status="succeeded",
        best_win_rate=0.51,
        download_available=True,
    )
    snapshots = []

    result = run_autoscientist(
        client,
        state,
        on_run_started=lambda current: snapshots.append(current.to_dict()),
    )

    arguments = client.autoscientist.create_arguments
    assert arguments["hyperparams"] == {
        "learning_rate": 2e-4,
        "n_epochs": 3.0,
    }
    assert arguments["idempotency_key"].startswith("falsifyrl-v3-")
    assert arguments["idempotency_key"] != "falsifyrl-v2-training-456"
    assert snapshots[0]["autoscientist_run_ids"] == [
        "prior-run",
        "experiment-456",
    ]
    assert snapshots[0]["autoscientist_status"] is None
    assert snapshots[0]["best_win_rate"] is None
    assert snapshots[0]["download_available"] is False
    assert result.autoscientist_status == "succeeded"


def test_training_rejects_unreviewed_adapted_data() -> None:
    state = WorkflowState(
        plan=_plan(),
        dataset_id="dataset-123",
        dataset_status="succeeded",
    )
    with pytest.raises(ValueError, match="exported and audited"):
        run_autoscientist(FakeClient(), state)


def test_export_audits_exact_adapted_dataset(tmp_path) -> None:
    source = tmp_path / "source.csv"
    diagnosis = (
        '{"confidence":0.99,"counterexample_config":{"behavior_profile":"safe"},'
        '"evidence_steps":[],"expected_effect":"No patch needed.",'
        '"failure_type":"none","responsible_agents":[],"reward_patch":null,'
        '"verdict":"aligned"}'
    )
    with source.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=("prompt", "completion"))
        writer.writeheader()
        writer.writerow({"prompt": "exact prompt", "completion": diagnosis})

    adapted_buffer = io.StringIO(newline="")
    writer = csv.DictWriter(
        adapted_buffer,
        fieldnames=(
            "original_prompt",
            "original_completion",
            "enhanced_prompt",
            "enhanced_completion",
        ),
    )
    writer.writeheader()
    writer.writerow(
        {
            "original_prompt": "exact prompt",
            "original_completion": diagnosis,
            "enhanced_prompt": "exact prompt",
            "enhanced_completion": diagnosis.replace("0.99", "0.98"),
        }
    )

    state = WorkflowState(
        plan=_plan(),
        dataset_id="dataset-123",
        dataset_status="succeeded",
        adaptation_run_id="adaptation-789",
    )
    result = export_and_audit_adapted_dataset(
        FakeClient(),
        state,
        tmp_path / "adapted.csv",
        source,
        fetcher=lambda url: adapted_buffer.getvalue().encode("utf-8"),
    )

    assert result.adapted_schema_valid is True
    assert result.adapted_row_count == 1
    assert result.adapted_prompt_column == "enhanced_prompt"
    assert result.adapted_completion_column == "enhanced_completion"
    assert result.adapted_export_sha256
    assert result.adapted_audit_sha256


def test_export_accepts_csv_text_returned_by_sdk(tmp_path) -> None:
    source = tmp_path / "source.csv"
    diagnosis = (
        '{"confidence":0.99,"counterexample_config":{"behavior_profile":"safe"},'
        '"evidence_steps":[],"expected_effect":"No patch needed.",'
        '"failure_type":"none","responsible_agents":[],"reward_patch":null,'
        '"verdict":"aligned"}'
    )
    with source.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=("prompt", "completion"))
        writer.writeheader()
        writer.writerow({"prompt": "exact prompt", "completion": diagnosis})
        writer.writerow({"prompt": "exact prompt", "completion": diagnosis})

    adapted_buffer = io.StringIO(newline="")
    writer = csv.DictWriter(
        adapted_buffer,
        fieldnames=(
            "prompt",
            "completion",
            "enhanced_prompt",
            "enhanced_completion",
        ),
    )
    writer.writeheader()
    writer.writerow(
        {
            "prompt": "exact prompt",
            "completion": diagnosis,
            "enhanced_prompt": "",
            "enhanced_completion": "not strict JSON",
        }
    )
    client = FakeClient()
    client.datasets.download = lambda dataset_id, file_format: adapted_buffer.getvalue()
    state = WorkflowState(
        plan=AutoScientistPlan(
            source="huggingface",
            source_url="https://huggingface.co/datasets/example/falsifyrl",
            expected_training_rows=2,
        ),
        dataset_id="dataset-123",
        dataset_status="succeeded",
        adaptation_run_id="adaptation-789",
    )

    result = export_and_audit_adapted_dataset(
        client,
        state,
        tmp_path / "adapted.csv",
        source,
    )

    assert result.adapted_schema_valid is True
    assert result.adapted_row_count == 1
    assert result.adapted_prompt_column == "prompt"
    assert result.adapted_completion_column == "completion"
    audit = json.loads(
        (tmp_path / "adapted.audit.json").read_text(encoding="utf-8")
    )
    assert audit["source_row_count"] == 2
    assert audit["source_unique_row_count"] == 1
    assert audit["exact_duplicate_rows_collapsed"] == 1
    assert audit["rejected_completion_columns"] == ["enhanced_completion"]


def test_training_accepts_audited_exact_duplicate_collapse() -> None:
    state = WorkflowState(
        plan=AutoScientistPlan(
            source="huggingface",
            source_url="https://huggingface.co/datasets/example/falsifyrl",
            expected_training_rows=2,
        ),
        dataset_id="dataset-123",
        dataset_status="succeeded",
        adapted_export_sha256="a" * 64,
        adapted_audit_sha256="b" * 64,
        adapted_row_count=1,
        adapted_prompt_column="prompt",
        adapted_completion_column="completion",
        adapted_schema_valid=True,
        training_dataset_id="training-456",
        training_dataset_status="succeeded",
        training_prompt_column="original_prompt",
        training_completion_column="original_completion",
    )

    result = run_autoscientist(FakeClient(), state)

    assert result.autoscientist_status == "succeeded"


def test_prepare_training_dataset_uses_exact_passthrough_export(tmp_path) -> None:
    export = tmp_path / "adapted.csv"
    export.write_text(
        "prompt,completion,enhanced_completion\n"
        "p1,c1,invalid\n"
        "p2,c2,invalid\n",
        encoding="utf-8",
    )
    state = WorkflowState(
        plan=_plan(),
        dataset_id="dataset-123",
        dataset_status="succeeded",
        adapted_export_path=str(export),
        adapted_export_sha256=hashlib.sha256(export.read_bytes()).hexdigest(),
        adapted_audit_sha256="b" * 64,
        adapted_row_count=2,
        adapted_prompt_column="prompt",
        adapted_completion_column="completion",
        adapted_schema_valid=True,
    )
    snapshots = []
    client = FakeClient()

    result = prepare_training_dataset(
        client,
        state,
        on_dataset_created=lambda current: snapshots.append(current.to_dict()),
    )

    assert result.training_dataset_id == "training-456"
    assert result.training_dataset_status == "succeeded"
    assert result.training_prompt_column == "original_prompt"
    assert result.training_completion_column == "original_completion"
    assert snapshots[0]["training_dataset_id"] == "training-456"
    assert client.datasets.upload_calls[0]["processing_mode"] == "passthrough"
    assert client.datasets.upload_calls[0]["column_mapping"] == {
        "prompt": "prompt",
        "completion": "completion",
    }


def test_export_rejects_non_https_download_url(tmp_path) -> None:
    client = FakeClient()
    client.datasets.download = lambda dataset_id, file_format: (
        "http://example.test/adapted.csv"
    )
    state = WorkflowState(
        plan=_plan(),
        dataset_id="dataset-123",
        dataset_status="succeeded",
        adaptation_run_id="adaptation-789",
    )

    with pytest.raises(ValueError, match="non-HTTPS"):
        export_and_audit_adapted_dataset(
            client,
            state,
            tmp_path / "adapted.csv",
            tmp_path / "source.csv",
        )


def test_adaptation_persists_run_id_before_waiting() -> None:
    client = FakeClient()
    state = WorkflowState(
        plan=_plan(),
        dataset_id="dataset-123",
        dataset_status="awaiting_input",
    )
    snapshots = []

    result = run_adaptation(
        client,
        state,
        on_adaptation_started=lambda current: snapshots.append(current.to_dict()),
    )

    assert result.dataset_status == "succeeded"
    assert result.adaptation_run_id is not None
    assert snapshots[0]["adaptation_run_id"] == result.adaptation_run_id
    adaptation_call = client.datasets.run_calls[0]
    assert adaptation_call["job_specification"]["idempotency_key"] == (
        "falsifyrl-adapt-v1-dataset-123"
    )


def test_resume_waits_on_existing_adaptation_without_starting_another(tmp_path) -> None:
    class ExistingDatasets:
        def get(self, dataset_id):
            assert dataset_id == "dataset-123"
            return SimpleNamespace(
                status="succeeded",
                progress=SimpleNamespace(processed_rows=1, total_rows=1),
                error_data=None,
            )

        def run(self, *args, **kwargs):
            raise AssertionError("resume must not create another dataset run")

    state_path = tmp_path / "workflow.json"
    state = WorkflowState(
        plan=_plan(),
        dataset_id="dataset-123",
        dataset_status="running",
        adaptation_run_id="adaptation-789",
    )
    state.save(state_path)

    result = await_existing_adaptation(
        SimpleNamespace(datasets=ExistingDatasets()),
        state,
        state_path,
        poll_seconds=0,
        timeout_seconds=1,
    )

    assert result.dataset_status == "succeeded"
    assert WorkflowState.load(state_path).dataset_status == "succeeded"
