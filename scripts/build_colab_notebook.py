from __future__ import annotations

import json
from pathlib import Path

if __package__:
    from .build_kaggle_notebook import notebook
else:
    from build_kaggle_notebook import notebook


def colab_notebook() -> dict:
    value = notebook()
    cells = value["cells"]
    cells[0]["source"] = [
        line
        for line in """# FalsifyRL — Colab GPU held-out evaluation

This notebook evaluates the unadapted base model and public AutoScientist LoRA on the entirely
held-out `crossing_navigation` family. Select a paid Colab L4/A100 runtime before running all cells.
Set `FALSIFYRL_MAX_EXAMPLES` for a smoke test; omit it for the exact 640-example comparison.
""".splitlines(keepends=True)
    ]
    cells[1]["source"] = [
        line
        for line in (
            '%pip install -q "transformers>=5.8,<6" "peft>=0.17,<1" '
            '"accelerate>=1,<2"\n'
            '%pip install -q "pillow>=11,<13" "huggingface_hub>=0.36,<2"\n'
        ).splitlines(keepends=True)
    ]
    cells[2]["source"] = [
        line
        for line in """import json
import os
from collections import Counter, defaultdict
from pathlib import Path

from huggingface_hub import hf_hub_download

DATASET_REPO_ID = os.environ.get(
    "FALSIFYRL_DATASET_REPO_ID",
    "KuanKuanKuan/falsifyrl-adapted",
)
TEST_PATH = Path(hf_hub_download(
    repo_id=DATASET_REPO_ID,
    filename="test.jsonl",
    repo_type="dataset",
))
rows = [json.loads(line) for line in TEST_PATH.read_text().splitlines() if line.strip()]
print("test path:", TEST_PATH)
print("examples:", len(rows), "roles:", Counter(row["case_role"] for row in rows))
assert len(rows) == 640
assert {row["scenario_family"] for row in rows} == {"crossing_navigation"}
""".splitlines(keepends=True)
    ]
    cells[5]["source"] = [
        line
        for line in """## Load the public AutoScientist adapter

The Hugging Face adapter config names the exact base model selected by AutoScientist.
Override `FALSIFYRL_MODEL_REPO_ID` only when auditing a different public release.
""".splitlines(keepends=True)
    ]
    cells[6]["source"] = [
        line
        for line in """import torch
from huggingface_hub import snapshot_download
from peft import PeftModel
from transformers import AutoModelForMultimodalLM, AutoProcessor

MODEL_REPO_ID = os.environ.get(
    "FALSIFYRL_MODEL_REPO_ID",
    "KuanKuanKuan/falsifyrl-autoscientist",
)
ADAPTER_DIR = Path(snapshot_download(MODEL_REPO_ID))
adapter_config = json.loads((ADAPTER_DIR / "adapter_config.json").read_text())
BASE_MODEL_ID = adapter_config["base_model_name_or_path"]
print("adapter:", MODEL_REPO_ID)
print("base model:", BASE_MODEL_ID)

processor = AutoProcessor.from_pretrained(BASE_MODEL_ID)
processor.tokenizer.padding_side = "left"
base_model = AutoModelForMultimodalLM.from_pretrained(
    BASE_MODEL_ID,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    device_map="auto",
)
base_model.eval()
""".splitlines(keepends=True)
    ]
    cells[9]["source"] = [
        line
        for line in "".join(cells[9]["source"])
        .replace("/kaggle/working/", "/content/")
        .splitlines(keepends=True)
    ]
    cells[10]["source"] = [
        line
        for line in """Download both prediction JSONL files from the Colab file browser. The public
Kaggle notebook is the canonical reproducible run; this Colab version is an accelerator fallback
using the same dataset, model, deterministic decoding, and 640 held-out examples.
""".splitlines(keepends=True)
    ]
    return value


def main() -> None:
    destination = Path("colab/falsifyrl_evaluation.ipynb")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(colab_notebook(), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(destination.resolve())


if __name__ == "__main__":
    main()
