from __future__ import annotations

from scripts.generate_model_predictions import extract_json_object


def test_extract_json_object_strips_model_wrapping_text() -> None:
    assert extract_json_object('answer: {"verdict":"aligned"} done') == (
        '{"verdict":"aligned"}'
    )
    assert extract_json_object("not json") == "not json"
