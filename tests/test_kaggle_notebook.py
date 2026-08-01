from __future__ import annotations

import json
from pathlib import Path

from scripts.build_colab_notebook import colab_notebook
from scripts.build_kaggle_notebook import notebook
from scripts.continue_kaggle_notebook import (
    await_verified_model_release,
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


def test_kaggle_release_declares_exact_adapted_dataset_and_model(
    tmp_path: Path,
) -> None:
    output = tmp_path / "release"
    metadata = prepare_kaggle_notebook(
        owner="owner",
        notebook_path=Path("kaggle/falsifyrl_evaluation.ipynb"),
        metadata_template=Path("kaggle/kernel-metadata.template.json"),
        output_dir=output,
        model_version=3,
    )

    assert metadata["dataset_sources"] == ["owner/falsifyrl-adapted"]
    assert metadata["model_sources"] == [
        "owner/falsifyrl-autoscientist/pytorch/lora/3"
    ]
    assert (output / "falsifyrl_evaluation.ipynb").is_file()
    assert (output / "kernel-metadata.json").is_file()


def test_colab_notebook_uses_public_huggingface_artifacts() -> None:
    value = colab_notebook()
    rendered = "\n".join(
        "".join(cell["source"])
        for cell in value["cells"]
    )

    assert "KuanKuanKuan/falsifyrl-adapted" in rendered
    assert "KuanKuanKuan/falsifyrl-autoscientist" in rendered
    assert "hf_hub_download" in rendered
    assert "snapshot_download" in rendered
    assert "falsifyrl-autoscientist-current-checkpoint.tar.zst" in rendered
    assert 'drive.mount("/content/drive")' in rendered
    assert 'userdata.get("HF_TOKEN")' in rendered
    assert '"Qwen/Qwen3.5-9B"' in rendered
    assert "/content/drive/MyDrive/FalsifyRL/evaluation" in rendered
    assert "2f10c842-c124-407b-89c0-f4af5a761bb4" in rendered
    assert "/kaggle/" not in rendered


def test_colab_notebook_can_pin_a_backup_run() -> None:
    value = colab_notebook(
        base_model_id="meta-llama/Llama-3.2-3B-Instruct",
        run_id="llama-run",
        archive_name="llama-checkpoint.tar.zst",
    )
    rendered = "\n".join("".join(cell["source"]) for cell in value["cells"])

    assert "meta-llama/Llama-3.2-3B-Instruct" in rendered
    assert "llama-run" in rendered
    assert "llama-checkpoint.tar.zst" in rendered
    assert '"Qwen/Qwen3.5-9B"' not in rendered


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
