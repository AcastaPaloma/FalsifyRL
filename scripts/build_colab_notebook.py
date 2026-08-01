from __future__ import annotations

import argparse
import json
from pathlib import Path

if __package__:
    from .build_kaggle_notebook import notebook
else:
    from build_kaggle_notebook import notebook


DEFAULT_BASE_MODEL_ID = "Qwen/Qwen3.5-9B"
DEFAULT_RUN_ID = "2f10c842-c124-407b-89c0-f4af5a761bb4"
DEFAULT_ARCHIVE_NAME = "falsifyrl-autoscientist-current-checkpoint.tar.zst"


def colab_notebook(
    *,
    base_model_id: str = DEFAULT_BASE_MODEL_ID,
    run_id: str = DEFAULT_RUN_ID,
    archive_name: str = DEFAULT_ARCHIVE_NAME,
) -> dict:
    value = notebook()
    cells = value["cells"]
    cells[0]["source"] = [
        line
        for line in """# FalsifyRL — Colab GPU held-out evaluation

This notebook evaluates the unadapted base model and the exact AutoScientist LoRA on the entirely
held-out `crossing_navigation` family. It can load a private, pre-publication checkpoint from your
Google Drive or the public Hugging Face release. Select a paid Colab L4/A100 runtime before running
all cells. Set `FALSIFYRL_MAX_EXAMPLES` for a smoke test; omit it for the exact 640-example
comparison. Add `HF_TOKEN` in Colab Secrets only when the selected base model is gated.
""".splitlines(keepends=True)
    ]
    cells[1]["source"] = [
        line
        for line in (
            '%pip install -q "transformers>=5.8,<6" "peft>=0.17,<1" '
            '"accelerate>=1,<2"\n'
            '%pip install -q "pillow>=11,<13" "huggingface_hub>=0.36,<2" '
            '"zstandard>=0.23,<1"\n'
        ).splitlines(keepends=True)
    ]
    cells[2]["source"] = [
        line
        for line in """import hashlib
import json
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
        for line in """## Load the exact AutoScientist adapter

Before publication, upload the downloaded AutoScientist checkpoint to Google Drive as
`falsifyrl-autoscientist-current-checkpoint.tar.zst`. The notebook mounts Drive and extracts it
locally. After publication, if that private file is absent, it falls back to the public Hugging Face
adapter. The expected base-model ID is pinned explicitly so internal training-provider aliases do
not leak into reproducibility.
""".splitlines(keepends=True)
    ]
    cells[6]["source"] = [
        line
        for line in """import shutil
import tarfile
import tempfile

import torch
import zstandard
from google.colab import drive, userdata
from huggingface_hub import snapshot_download
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoModelForMultimodalLM, AutoTokenizer

try:
    HF_TOKEN = userdata.get("HF_TOKEN")
except Exception:
    HF_TOKEN = os.environ.get("HF_TOKEN")

EXPECTED_BASE_MODEL_ID = os.environ.get(
    "FALSIFYRL_BASE_MODEL_ID",
    "Qwen/Qwen3.5-9B",
)
PRIVATE_ARCHIVE_NAME = os.environ.get(
    "FALSIFYRL_ADAPTER_ARCHIVE_NAME",
    "falsifyrl-autoscientist-current-checkpoint.tar.zst",
)

MODEL_REPO_ID = os.environ.get(
    "FALSIFYRL_MODEL_REPO_ID",
    "KuanKuanKuan/falsifyrl-autoscientist",
)

def extract_private_adapter(archive_path):
    destination = Path("/content/falsifyrl-private-adapter")
    if destination.exists():
        shutil.rmtree(destination)
    destination.mkdir(parents=True)
    with tempfile.NamedTemporaryFile(suffix=".tar") as decompressed:
        with Path(archive_path).open("rb") as source:
            reader = zstandard.ZstdDecompressor().stream_reader(source)
            shutil.copyfileobj(reader, decompressed)
        decompressed.flush()
        with tarfile.open(decompressed.name, mode="r:") as archive:
            archive.extractall(destination, filter="data")
    configs = list(destination.rglob("adapter_config.json"))
    assert len(configs) == 1, f"expected one adapter config, found {len(configs)}"
    adapter_dir = configs[0].parent
    assert (adapter_dir / "adapter_model.safetensors").is_file()
    return adapter_dir

drive.mount("/content/drive")
private_matches = list(Path("/content/drive/MyDrive").rglob(PRIVATE_ARCHIVE_NAME))
assert len(private_matches) <= 1, (
    f"found multiple private checkpoints named {PRIVATE_ARCHIVE_NAME}; keep exactly one"
)
if private_matches:
    ADAPTER_DIR = extract_private_adapter(private_matches[0])
    ADAPTER_SOURCE = str(private_matches[0])
else:
    ADAPTER_DIR = Path(snapshot_download(MODEL_REPO_ID, token=HF_TOKEN))
    ADAPTER_SOURCE = MODEL_REPO_ID

adapter_config = json.loads((ADAPTER_DIR / "adapter_config.json").read_text())
checkpoint_base = adapter_config["base_model_name_or_path"]
checkpoint_slug = checkpoint_base.lower().replace("reference", "").replace("__tog__ft", "")
expected_slug = EXPECTED_BASE_MODEL_ID.lower().split("/")[-1]
assert expected_slug in checkpoint_slug or checkpoint_slug.endswith(expected_slug), (
    f"checkpoint base {checkpoint_base!r} does not match {EXPECTED_BASE_MODEL_ID!r}"
)
BASE_MODEL_ID = EXPECTED_BASE_MODEL_ID
print("adapter:", ADAPTER_SOURCE)
print("base model:", BASE_MODEL_ID)

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID, token=HF_TOKEN)
tokenizer.padding_side = "left"
if tokenizer.pad_token_id is None:
    tokenizer.pad_token = tokenizer.eos_token
model_kwargs = {
    "token": HF_TOKEN,
    "torch_dtype": (
        torch.bfloat16
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
        else torch.float16 if torch.cuda.is_available() else torch.float32
    ),
    "device_map": "auto",
    "low_cpu_mem_usage": True,
}
try:
    base_model = AutoModelForCausalLM.from_pretrained(BASE_MODEL_ID, **model_kwargs)
except (TypeError, ValueError):
    base_model = AutoModelForMultimodalLM.from_pretrained(BASE_MODEL_ID, **model_kwargs)
base_model.eval()
""".splitlines(keepends=True)
    ]
    cells[9]["source"] = [
        line
        for line in """def save_predictions(path, predictions):
    with path.open("w") as stream:
        for row, completion in zip(
            rows[:MAX_EXAMPLES], predictions, strict=True
        ):
            stream.write(json.dumps({
                "example_id": row["example_id"],
                "completion": completion,
            }) + "\\n")

def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

RUN_ID = os.environ.get(
    "FALSIFYRL_RUN_ID",
    "2f10c842-c124-407b-89c0-f4af5a761bb4",
)
COLAB_OUTPUT_DIR = (
    Path("/content/drive/MyDrive/FalsifyRL/evaluation") / RUN_ID
)
COLAB_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
base_prediction_path = COLAB_OUTPUT_DIR / "falsifyrl-base-test-predictions.jsonl"
adapted_prediction_path = COLAB_OUTPUT_DIR / "falsifyrl-adapted-test-predictions.jsonl"
save_predictions(base_prediction_path, base_predictions)
save_predictions(adapted_prediction_path, adapted_predictions)

adapter_weights = ADAPTER_DIR / "adapter_model.safetensors"
report = {
    "run_id": RUN_ID,
    "dataset_test_path": str(TEST_PATH),
    "adapter_path": str(ADAPTER_DIR),
    "adapter_sha256": sha256(adapter_weights),
    "base_model_id": BASE_MODEL_ID,
    "example_count": MAX_EXAMPLES,
    "base_predictions_sha256": sha256(base_prediction_path),
    "adapted_predictions_sha256": sha256(adapted_prediction_path),
    "base_metrics": base_metrics,
    "adapted_metrics": adapted_metrics,
    "improvement": {
        key: adapted_metrics[key] - base_metrics[key]
        for key in adapted_metrics
        if isinstance(adapted_metrics[key], float)
    },
}
report_path = COLAB_OUTPUT_DIR / "colab-evaluation.json"
report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\\n")
print(json.dumps(report, indent=2, sort_keys=True))
print("saved:", COLAB_OUTPUT_DIR)
""".splitlines(keepends=True)
    ]
    cells[10]["source"] = [
        line
        for line in """Download both prediction JSONL files from the Colab file browser. The
repository's CPU-only evaluator validates the complete output schema and re-executes every proposed
reward patch.
This Colab notebook performs only the GPU-heavy base and adapter inference over the exact same 640
held-out examples.
""".splitlines(keepends=True)
    ]
    replacements = {
        DEFAULT_BASE_MODEL_ID: base_model_id,
        DEFAULT_RUN_ID: run_id,
        DEFAULT_ARCHIVE_NAME: archive_name,
    }
    for cell in cells:
        source = "".join(cell["source"])
        for original, replacement in replacements.items():
            source = source.replace(original, replacement)
        cell["source"] = source.splitlines(keepends=True)
    return value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a GPU-only Colab inference notebook for an exact run."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("colab/falsifyrl_evaluation.ipynb"),
    )
    parser.add_argument("--base-model-id", default=DEFAULT_BASE_MODEL_ID)
    parser.add_argument("--run-id", default=DEFAULT_RUN_ID)
    parser.add_argument("--archive-name", default=DEFAULT_ARCHIVE_NAME)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    destination = args.output
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(
            colab_notebook(
                base_model_id=args.base_model_id,
                run_id=args.run_id,
                archive_name=args.archive_name,
            ),
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    print(destination.resolve())


if __name__ == "__main__":
    main()
