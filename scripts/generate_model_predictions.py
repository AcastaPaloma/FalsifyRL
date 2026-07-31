from __future__ import annotations

import argparse
import json
from pathlib import Path

from falsifyrl.dataset import DatasetBuildConfig, build_cases


def extract_json_object(text: str) -> str:
    start = text.find("{")
    end = text.rfind("}")
    return text[start : end + 1] if start >= 0 and end >= start else text.strip()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate deterministic FalsifyRL base or PEFT-adapter predictions."
    )
    parser.add_argument("--model-id", default="Qwen/Qwen3.5-0.8B")
    parser.add_argument("--adapter", type=Path)
    parser.add_argument("--split", choices=("validation", "test"), default="test")
    parser.add_argument("--max-examples", type=int)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.max_examples is not None and args.max_examples < 1:
        raise ValueError("--max-examples must be positive")

    import torch
    from transformers import AutoModelForMultimodalLM, AutoProcessor

    processor = AutoProcessor.from_pretrained(args.model_id)
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
    with args.output.open("w", encoding="utf-8") as stream:
        for index, case in enumerate(cases, start=1):
            inputs = processor.apply_chat_template(
                [
                    {
                        "role": "user",
                        "content": [{"type": "text", "text": case.render_prompt()}],
                    }
                ],
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt",
            ).to(model.device)
            with torch.inference_mode():
                output = model.generate(
                    **inputs,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=False,
                    pad_token_id=processor.tokenizer.eos_token_id,
                )
            generated = processor.decode(
                output[0, inputs["input_ids"].shape[1] :],
                skip_special_tokens=True,
            )
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
            stream.flush()
            if index % 20 == 0 or index == len(cases):
                print(f"generated {index}/{len(cases)}")


if __name__ == "__main__":
    main()
