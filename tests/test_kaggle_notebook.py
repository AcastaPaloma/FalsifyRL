from __future__ import annotations

from scripts.build_kaggle_notebook import notebook


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
