from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.generate_model_predictions import (
    batches,
    completed_prefix_count,
    extract_json_object,
)


def test_extract_json_object_strips_model_wrapping_text() -> None:
    assert extract_json_object('answer: {"verdict":"aligned"} done') == (
        '{"verdict":"aligned"}'
    )
    assert extract_json_object("not json") == "not json"


def test_batches_preserves_order_and_rejects_invalid_size() -> None:
    assert list(batches([1, 2, 3, 4, 5], 2)) == [[1, 2], [3, 4], [5]]
    with pytest.raises(ValueError, match="positive"):
        list(batches([1], 0))


def test_completed_prefix_count_requires_exact_order(tmp_path: Path) -> None:
    path = tmp_path / "predictions.jsonl"
    path.write_text(
        json.dumps({"example_id": "a", "completion": "{}"}) + "\n",
        encoding="utf-8",
    )
    assert completed_prefix_count(path, ["a", "b"]) == 1

    path.write_text(
        json.dumps({"example_id": "b", "completion": "{}"}) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="exact expected prefix"):
        completed_prefix_count(path, ["a", "b"])
