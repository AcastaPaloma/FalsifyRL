from __future__ import annotations

import json
from pathlib import Path

from falsifyrl.demo import (
    parse_prompt_sections,
    prepare_space_bundle,
    select_demo_examples,
    trace_table,
)


def _record(example_id: str, failure: str, role: str) -> dict:
    prompt = """Instruction.

SCENARIO FAMILY:
crossing_navigation

TASK SPECIFICATION:
Both robots reach their goals safely.

PROXY REWARD PROGRAM:
def reward(): return 1

OBSERVED EPISODE TRACE:
step | agent_a_progress | agent_b_progress
0 | 0.5 | 0.4
"""
    return {
        "example_id": example_id,
        "pair_id": "pair-1",
        "case_role": role,
        "failure_type": failure,
        "prompt": prompt,
        "completion": "{}",
    }


def test_prompt_parser_and_trace_table() -> None:
    sections = parse_prompt_sections(_record("a", "none", "control")["prompt"])
    rows = trace_table(sections["episode_trace"])

    assert sections["scenario_family"] == "crossing_navigation"
    assert rows[0] == ["step", "agent_a_progress", "agent_b_progress"]
    assert rows[1] == ["0", "0.5", "0.4"]


def test_demo_selection_keeps_one_example_per_failure_and_role(
    tmp_path: Path,
) -> None:
    dataset = tmp_path / "test.jsonl"
    records = [
        _record("a", "collision_blind", "control"),
        _record("b", "collision_blind", "exploit"),
        {**_record("c", "collision_blind", "exploit"), "pair_id": "pair-2"},
    ]
    dataset.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )

    selected = select_demo_examples(dataset)

    assert [record["example_id"] for record in selected] == ["a", "b"]
    assert {record["pair_failure_type"] for record in selected} == {"collision_blind"}


def test_space_bundle_contains_no_training_metadata_beyond_examples(
    tmp_path: Path,
) -> None:
    template = tmp_path / "space"
    template.mkdir()
    for filename in ("README.md", "app.py", "requirements.txt"):
        (template / filename).write_text(filename, encoding="utf-8")
    dataset = tmp_path / "test.jsonl"
    dataset.write_text(
        json.dumps(_record("a", "none", "control"))
        + "\n"
        + json.dumps(_record("b", "collision_blind", "exploit"))
        + "\n",
        encoding="utf-8",
    )

    summary = prepare_space_bundle(template, dataset, tmp_path / "bundle")

    assert summary["example_count"] == 2
    assert summary["roles"] == ["control", "exploit"]
    assert (tmp_path / "bundle" / "examples.json").is_file()


def test_space_source_has_valid_unicode_and_oracle_summary() -> None:
    app = Path("space/app.py").read_text(encoding="utf-8")
    card = Path("space/README.md").read_text(encoding="utf-8")

    assert "✅ Aligned control trace" in app
    assert "⚠️ Counterexample trace" in app
    assert "Independent task oracle" in app
    assert "emoji: 🔬" in card
    assert not any(marker in app + card for marker in ("â", "ð", "Â", "ï"))
