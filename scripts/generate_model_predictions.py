from __future__ import annotations

import argparse
import json
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import TypeVar

from falsifyrl.dataset import DatasetBuildConfig, build_cases

T = TypeVar("T")


def extract_json_object(text: str) -> str:
    start = text.find("{")
    end = text.rfind("}")
    return text[start : end + 1] if start >= 0 and end >= start else text.strip()


def batches(values: Sequence[T], size: int) -> Iterator[Sequence[T]]:
    if size < 1:
        raise ValueError("batch size must be positive")
    for start in range(0, len(values), size):
        yield values[start : start + size]


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
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.max_examples is not None and args.max_examples < 1:
        raise ValueError("--max-examples must be positive")
    if args.batch_size < 1:
        raise ValueError("--batch-size must be positive")

    import torch
    from transformers import AutoModelForMultimodalLM, AutoProcessor

    processor = AutoProcessor.from_pretrained(args.model_id)
    processor.tokenizer.padding_side = "left"
    model = AutoModelForMultimodalLM.from_pretrained(
        args.model_id,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
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
    generated_count = 0
    with args.output.open("w", encoding="utf-8") as stream:
        for case_batch in batches(cases, args.batch_size):
            formatted = [
                processor.apply_chat_template(
                    [
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": case.render_prompt()}
                            ],
                        }
                    ],
                    tokenize=False,
                    add_generation_prompt=True,
                )
                for case in case_batch
            ]
            inputs = processor(
                text=formatted,
                padding=True,
                return_tensors="pt",
            ).to(model.device)
            with torch.inference_mode():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=False,
                    pad_token_id=processor.tokenizer.eos_token_id,
                )
            input_length = inputs["input_ids"].shape[1]
            decoded = processor.batch_decode(
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
