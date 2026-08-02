from __future__ import annotations

import json
import subprocess
from pathlib import Path

from scripts import continue_kaggle_notebook
from scripts.build_colab_notebook import colab_notebook
from scripts.build_kaggle_notebook import notebook
from scripts.continue_kaggle_notebook import (
    await_kernel_completion,
    await_verified_model_release,
    parse_kernel_status,
    update_private_manifest,
)
from scripts.prepare_kaggle_notebook import prepare_kaggle_notebook


def test_kaggle_notebook_contains_reproducibility_contract() -> None:
    value = notebook()
    rendered = "\n".join(
        "".join(cell["source"])
        for cell in value["cells"]
    )

    assert value["nbformat"] == 4
    assert "crossing_navigation" in rendered
    assert "assert len(rows) == 640" in rendered
    assert "adapter_config.json" in rendered
    assert "AutoModelForMultimodalLM" in rendered
    assert "AutoModelForCausalLM" in rendered
    assert "AutoTokenizer" in rendered
    assert '"role": "user", "content": prompt' in rendered
    assert "FALSIFYRL_MAX_EXAMPLES" in rendered
    assert "falsifyrl-base-test-predictions.jsonl" in rendered
    assert "falsifyrl-adapted-test-predictions.jsonl" in rendered
    assert '"base_metrics": base_metrics' in rendered
    assert '"adapted_metrics": adapted_metrics' in rendered
    assert '"improvement"' in rendered
    assert "executable-patch metric" in rendered
    assert 'FALSIFYRL_BATCH_SIZE", 1' in rendered
    assert "FALSIFYRL_MAX_NEW_TOKENS" in rendered
    assert "tokenizer.batch_decode" in rendered
    assert 'UserSecretsClient().get_secret("HF_TOKEN")' in rendered
    assert "Llama-FalsifyRL-AutoScientist/pytorch/lora" in rendered
    assert "commit_verified_colab_evidence" in rendered
    assert "load_predictions(BASE_PREDICTION_SOURCE)" in rendered
    assert 'RELEASE_MANIFEST["files"][prediction_path.name]["sha256"]' in rendered


def test_kaggle_release_declares_exact_adapted_dataset_and_model(
    tmp_path: Path,
) -> None:
    output = tmp_path / "release"
    metadata = prepare_kaggle_notebook(
        owner="owner",
        notebook_path=Path("kaggle/falsifyrl_evaluation.ipynb"),
        metadata_template=Path("kaggle/kernel-metadata.template.json"),
        output_dir=output,
        model_slug="Llama-FalsifyRL-AutoScientist",
        model_version=3,
    )

    assert metadata["dataset_sources"] == ["owner/falsifyrl-adapted"]
    assert metadata["model_sources"] == [
        "owner/Llama-FalsifyRL-AutoScientist/pytorch/lora/3"
    ]
    assert (output / "falsifyrl_evaluation.ipynb").is_file()
    assert (output / "kernel-metadata.json").is_file()


def test_colab_notebook_uses_private_commit_pinned_huggingface_staging() -> None:
    value = colab_notebook(
        staging_repo_id="owner/private-eval-staging",
        staging_revision="a" * 40,
        staging_adapter_path="runs/qwen-run/adapter",
    )
    rendered = "\n".join(
        "".join(cell["source"])
        for cell in value["cells"]
    )

    assert "KuanKuanKuan/falsifyrl-adapted" in rendered
    assert "KuanKuanKuan/falsifyrl-autoscientist" in rendered
    assert "hf_hub_download" in rendered
    assert "snapshot_download" in rendered
    assert "owner/private-eval-staging" in rendered
    assert "a" * 40 in rendered
    assert "runs/qwen-run/adapter" in rendered
    assert 'drive.mount("/content/drive")' not in rendered
    assert 'userdata.get("HF_TOKEN")' in rendered
    assert '"Qwen/Qwen3.5-9B"' in rendered
    assert "/content/FalsifyRL/evaluation" in rendered
    assert "2f10c842-c124-407b-89c0-f4af5a761bb4" in rendered
    assert "checkpoint-manifest.json" in rendered
    assert "evaluation-manifest.json" in rendered
    assert "parent_commit=current_head" in rendered
    assert "CommitOperationAdd" in rendered
    assert "%pip uninstall -q -y torchao" in rendered
    assert "/kaggle/" not in rendered


def test_colab_notebook_can_pin_a_backup_run() -> None:
    value = colab_notebook(
        base_model_id="meta-llama/Llama-3.2-3B-Instruct",
        run_id="llama-run",
        archive_name="llama-checkpoint.tar.zst",
        staging_repo_id="owner/private-stage",
        staging_revision="b" * 40,
        staging_adapter_path="runs/llama-run/adapter",
    )
    rendered = "\n".join("".join(cell["source"]) for cell in value["cells"])

    assert "meta-llama/Llama-3.2-3B-Instruct" in rendered
    assert "llama-run" in rendered
    assert "owner/private-stage" in rendered
    assert "b" * 40 in rendered
    assert "runs/llama-run/adapter" in rendered
    assert '"Qwen/Qwen3.5-9B"' not in rendered


def test_colab_notebook_optionally_uses_fail_closed_4bit_inference() -> None:
    value = colab_notebook(use_4bit=True)
    install_cell = "".join(value["cells"][1]["source"])
    model_cell = "".join(value["cells"][6]["source"])

    assert 'FALSIFYRL_USE_4BIT"] = "true"' in install_cell
    assert '"bitsandbytes>=0.46,<1"' in install_cell
    assert "BitsAndBytesConfig" in model_cell
    assert "4-bit inference requires a Colab GPU runtime" in model_cell
    assert "load_in_4bit=True" in model_cell
    assert 'bnb_4bit_quant_type="nf4"' in model_cell
    assert "bnb_4bit_use_double_quant=True" in model_cell
    assert "bnb_4bit_compute_dtype" in model_cell


def test_colab_notebook_keeps_4bit_opt_in_by_default() -> None:
    value = colab_notebook()
    install_cell = "".join(value["cells"][1]["source"])

    assert 'FALSIFYRL_USE_4BIT"] = "false"' in install_cell
    assert '"bitsandbytes>=0.46,<1"' not in install_cell
    assert 'os.environ.pop("FALSIFYRL_MAX_EXAMPLES", None)' in install_cell


def test_kaggle_run_waits_for_hash_verified_model_release(tmp_path: Path) -> None:
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "links": {
                    "kaggle_dataset": "https://www.kaggle.com/datasets/owner/data",
                    "kaggle_model": (
                        "https://www.kaggle.com/models/owner/model/pytorch/lora"
                    ),
                    "kaggle_notebook": None,
                },
                "attestations": {"weights_public_on_both_platforms": True},
            }
        ),
        encoding="utf-8",
    )

    manifest = await_verified_model_release(
        manifest_path,
        poll_seconds=0,
        timeout_seconds=1,
    )
    update_private_manifest(
        manifest_path,
        "https://www.kaggle.com/code/owner/falsifyrl-held-out-evaluation",
    )
    updated = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert manifest["attestations"]["weights_public_on_both_platforms"] is True
    assert updated["links"]["kaggle_notebook"].endswith(
        "falsifyrl-held-out-evaluation"
    )


def test_kaggle_kernel_status_is_polled_until_complete(
    tmp_path: Path,
    monkeypatch,
) -> None:
    statuses = iter(("Kernel status: running", "Kernel status: complete"))

    def fake_run(command, *, cwd):
        return subprocess.CompletedProcess(command, 0, next(statuses), "")

    monkeypatch.setattr(continue_kaggle_notebook, "_run", fake_run)
    monkeypatch.setattr(continue_kaggle_notebook.time, "sleep", lambda _: None)

    completed = await_kernel_completion(
        kaggle_cli=tmp_path / "kaggle.exe",
        kernel_handle="owner/kernel",
        repository=tmp_path,
        poll_seconds=0,
        timeout_seconds=1,
    )

    assert parse_kernel_status(completed.stdout) == "complete"
    assert parse_kernel_status("Kernel status: failed") == "failed"
    assert parse_kernel_status("unexpected output") == "unknown"
