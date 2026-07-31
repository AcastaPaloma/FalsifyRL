from __future__ import annotations

import pytest

from scripts.generate_model_predictions import batches, extract_json_object


def test_extract_json_object_strips_model_wrapping_text() -> None:
    assert extract_json_object('answer: {"verdict":"aligned"} done') == (
        '{"verdict":"aligned"}'
    )
    assert extract_json_object("not json") == "not json"


def test_batches_preserves_order_and_rejects_invalid_size() -> None:
    assert list(batches([1, 2, 3, 4, 5], 2)) == [[1, 2], [3, 4], [5]]
    with pytest.raises(ValueError, match="positive"):
        list(batches([1], 0))
