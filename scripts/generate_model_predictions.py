from __future__ import annotations

import argparse
import json
import os
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import TypeVar

from falsifyrl.dataset import DatasetBuildConfig, build_cases
from falsifyrl.evaluation import canonicalize_schema_aliases

T = TypeVar("T")


def extract_json_object(text: str) -> str:
    decoder = json.JSONDecoder()
    candidates: list[dict] = []
    for index, character in enumerate(text):
        if character != "{":
            continue
        try:
            value, _ = decoder.raw_decode(text, index)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            candidates.append(value)
    preferred = [
        value
        for value in candidates
        if {"verdict", "failure_type"}.issubset(value)
    ]
    if preferred or candidates:
        completion = json.dumps(
            (preferred or candidates)[-1],
            separators=(",", ":"),
            sort_keys=True,
        )
        return canonicalize_schema_aliases(completion)[0]
    return text.strip()


def batches(values: Sequence[T], size: int) -> Iterator[Sequence[T]]:
    if size < 1:
        raise ValueError("batch size must be positive")
    for start in range(0, len(values), size):
        yield values[start : start + size]


def completed_prefix_count(path: Path, expected_ids: Sequence[str]) -> int:
    if not path.exists():
        return 0
    completed = 0
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            row = json.loads(line)
            if set(row) != {"example_id", "completion"}:
                raise ValueError(
                    f"resume row {line_number} must contain example_id and completion"
                )
            if completed >= len(expected_ids):
                raise ValueError("resume file contains more rows than the selected split")
            if row["example_id"] != expected_ids[completed]:
                raise ValueError(
                    f"resume row {line_number} is not the exact expected prefix"
                )
            if not isinstance(row["completion"], str):
                raise ValueError(f"resume row {line_number} completion must be a string")
            completed += 1
    return completed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate deterministic FalsifyRL base or PEFT-adapter predictions."
    )
    parser.add_argument("--model-id", default="Qwen/Qwen3.5-0.8B")
    parser.add_argument("--adapter", type=Path)
    parser.add_argument("--split", choices=("validation", "test"), default="test")
    parser.add_argument("--max-examples", type=int)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.max_examples is not None and args.max_examples < 1:
        raise ValueError("--max-examples must be positive")
    if args.batch_size < 1:
        raise ValueError("--batch-size must be positive")

    import torch
    from transformers import (
        AutoModelForCausalLM,
        AutoModelForMultimodalLM,
        AutoTokenizer,
    )

    token = os.environ.get("HF_TOKEN")
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, token=token)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model_arguments = {
        "token": token,
        "torch_dtype": (
            torch.bfloat16
            if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
            else torch.float16 if torch.cuda.is_available() else torch.float32
        ),
        "device_map": "auto",
        "low_cpu_mem_usage": True,
    }
    try:
        model = AutoModelForCausalLM.from_pretrained(
            args.model_id,
            **model_arguments,
        )
    except (TypeError, ValueError):
        model = AutoModelForMultimodalLM.from_pretrained(
            args.model_id,
            **model_arguments,
        )
    if args.adapter is not None:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, args.adapter)
    model.eval()

    cases = [
        case
        for case in build_cases(DatasetBuildConfig())
        if case.scenario.split.value == args.split
    ]
    if args.max_examples is not None:
        cases = cases[: args.max_examples]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    generated_count = (
        completed_prefix_count(
            args.output,
            [case.example_id for case in cases],
        )
        if args.resume
        else 0
    )
    file_mode = "a" if args.resume and generated_count else "w"
    with args.output.open(file_mode, encoding="utf-8") as stream:
        for case_batch in batches(cases[generated_count:], args.batch_size):
            formatted = [
                tokenizer.apply_chat_template(
                    [
                        {
                            "role": "user",
                            "content": case.render_prompt(),
                        }
                    ],
                    tokenize=False,
                    add_generation_prompt=True,
                )
                for case in case_batch
            ]
            inputs = tokenizer(
                formatted,
                padding=True,
                return_tensors="pt",
            ).to(model.device)
            with torch.inference_mode():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id,
                )
            input_length = inputs["input_ids"].shape[1]
            decoded = tokenizer.batch_decode(
                outputs[:, input_length:],
                skip_special_tokens=True,
            )
            for case, generated in zip(case_batch, decoded, strict=True):
                stream.write(
                    json.dumps(
                        {
                            "example_id": case.example_id,
                            "completion": extract_json_object(generated),
                        },
                        sort_keys=True,
                    )
                    + "\n"
                )
            generated_count += len(case_batch)
            stream.flush()
            if generated_count % 20 == 0 or generated_count == len(cases):
                print(f"generated {generated_count}/{len(cases)}")


if __name__ == "__main__":
    main()
