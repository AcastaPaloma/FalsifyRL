from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any


def parse_prompt_sections(prompt: str) -> dict[str, str]:
    headings = {
        "SCENARIO FAMILY:": "scenario_family",
        "TASK SPECIFICATION:": "task_specification",
        "PROXY REWARD PROGRAM:": "reward_program",
        "OBSERVED EPISODE TRACE:": "episode_trace",
    }
    sections: dict[str, list[str]] = {"instruction": []}
    active = "instruction"
    for line in prompt.splitlines():
        if line in headings:
            active = headings[line]
            sections[active] = []
        else:
            sections.setdefault(active, []).append(line)
    return {
        name: "\n".join(lines).strip()
        for name, lines in sections.items()
    }


def trace_table(trace_text: str) -> list[list[str]]:
    lines = [line.strip() for line in trace_text.splitlines() if line.strip()]
    if len(lines) < 2:
        return []
    return [
        [cell.strip() for cell in line.split("|")]
        for line in lines
    ]


def select_demo_examples(
    dataset_jsonl: str | Path,
    *,
    maximum: int = 18,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with Path(dataset_jsonl).open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            records.append(
                {
                    "example_id": row["example_id"],
                    "pair_id": row["pair_id"],
                    "case_role": row["case_role"],
                    "failure_type": row["failure_type"],
                    "prompt": row["prompt"],
                    "gold_completion": row["completion"],
                }
            )

    pair_members: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        pair_members.setdefault(str(record["pair_id"]), []).append(record)
    selected_pairs: list[str] = []
    seen_failures: set[str] = set()
    for record in records:
        if record["case_role"] != "exploit":
            continue
        failure_type = str(record["failure_type"])
        if failure_type in seen_failures:
            continue
        seen_failures.add(failure_type)
        selected_pairs.append(str(record["pair_id"]))

    selected: list[dict[str, Any]] = []
    for pair_id in selected_pairs:
        members = sorted(
            pair_members[pair_id],
            key=lambda record: 0 if record["case_role"] == "control" else 1,
        )
        if len(selected) + len(members) > maximum:
            break
        pair_failure_type = next(
            str(record["failure_type"])
            for record in members
            if record["case_role"] == "exploit"
        )
        for member in members:
            member["pair_failure_type"] = pair_failure_type
        selected.extend(members)
    return selected


def prepare_space_bundle(
    template_dir: str | Path,
    dataset_jsonl: str | Path,
    bundle_dir: str | Path,
) -> dict[str, Any]:
    template = Path(template_dir)
    destination = Path(bundle_dir)
    destination.mkdir(parents=True, exist_ok=True)
    for filename in ("README.md", "app.py", "requirements.txt"):
        source = template / filename
        if not source.is_file():
            raise FileNotFoundError(source)
        shutil.copy2(source, destination / filename)

    examples = select_demo_examples(dataset_jsonl)
    (destination / "examples.json").write_text(
        json.dumps(examples, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return {
        "example_count": len(examples),
        "roles": sorted({example["case_role"] for example in examples}),
        "failure_types": sorted(
            {example["failure_type"] for example in examples}
        ),
    }
