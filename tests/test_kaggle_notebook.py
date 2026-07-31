from __future__ import annotations

from pathlib import Path

from scripts.build_kaggle_notebook import notebook
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
    assert "AutoProcessor" in rendered
    assert 'content": [{"type": "text"' in rendered
    assert "FALSIFYRL_MAX_EXAMPLES" in rendered
    assert "falsifyrl-base-test-predictions.jsonl" in rendered
    assert "falsifyrl-adapted-test-predictions.jsonl" in rendered
    assert '"base_metrics": base_metrics' in rendered
    assert '"adapted_metrics": adapted_metrics' in rendered
    assert '"improvement"' in rendered
    assert "executable-patch metric" in rendered


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
