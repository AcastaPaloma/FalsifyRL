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
STAGING_REPO_PLACEHOLDER = "__FALSIFYRL_STAGING_REPO_ID__"
STAGING_REVISION_PLACEHOLDER = "__FALSIFYRL_STAGING_REVISION__"
STAGING_ADAPTER_PATH_PLACEHOLDER = "__FALSIFYRL_STAGING_ADAPTER_PATH__"


def colab_notebook(
    *,
    base_model_id: str = DEFAULT_BASE_MODEL_ID,
    run_id: str = DEFAULT_RUN_ID,
    archive_name: str = DEFAULT_ARCHIVE_NAME,
    staging_repo_id: str | None = None,
    staging_revision: str | None = None,
    staging_adapter_path: str | None = None,
    max_examples: int | None = None,
    max_new_tokens: int = 768,
    batch_size: int = 1,
    use_4bit: bool = False,
) -> dict:
    value = notebook()
    cells = value["cells"]
    cells[0]["source"] = [
        line
        for line in """# FalsifyRL — Colab GPU held-out evaluation

This notebook evaluates the unadapted base model and the exact AutoScientist LoRA on the entirely
held-out `crossing_navigation` family. It loads either a commit-pinned private Hugging Face staging
checkpoint or the public Hugging Face release. Select a paid Colab L4/A100 runtime before running
all cells. Set `FALSIFYRL_MAX_EXAMPLES` for a smoke test; omit it for the exact 640-example
comparison. Add `HF_TOKEN` in Colab Secrets for private staging or a gated base model.
""".splitlines(keepends=True)
    ]
    cells[1]["source"] = [
        line
        for line in (
            '%pip install -q "transformers>=5.8,<6" "peft>=0.17,<1" '
            '"accelerate>=1,<2"\n'
            '%pip install -q "pillow>=11,<13" "huggingface_hub>=0.36,<2"\n'
            "%pip uninstall -q -y torchao\n"
        ).splitlines(keepends=True)
    ]
    runtime_config = [
        "import os\n",
        'os.environ.pop("FALSIFYRL_MAX_EXAMPLES", None)\n',
        f'os.environ["FALSIFYRL_MAX_NEW_TOKENS"] = "{max_new_tokens}"\n',
        f'os.environ["FALSIFYRL_BATCH_SIZE"] = "{batch_size}"\n',
        f'os.environ["FALSIFYRL_USE_4BIT"] = "{str(use_4bit).lower()}"\n',
    ]
    if max_examples is not None:
        runtime_config.append(
            f'os.environ["FALSIFYRL_MAX_EXAMPLES"] = "{max_examples}"\n'
        )
    cells[1]["source"] = [*runtime_config, "\n", *cells[1]["source"]]
    if use_4bit:
        cells[1]["source"].append(
            '%pip install -q "bitsandbytes>=0.46,<1"\n'
        )
    cells[1]["source"].extend(
        [
            "\n",
            "# Release stale model objects when rerunning a notebook in one runtime.\n",
            "import gc\n",
            'globals().pop("model", None)\n',
            'globals().pop("base_model", None)\n',
            "gc.collect()\n",
            "try:\n",
            "    import torch\n",
            "    torch.cuda.empty_cache()\n",
            "except Exception:\n",
            "    pass\n",
        ]
    )
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

Before publication, stage the exact AutoScientist checkpoint in a private Hugging Face model repo.
The notebook pins the immutable staging commit, verifies the checkpoint manifest and adapter hash,
then uploads only compact prediction evidence to the same private repo. After publication it can
instead use the public Hugging Face adapter. The expected base-model ID is pinned explicitly so
internal training-provider aliases do not leak into reproducibility.
""".splitlines(keepends=True)
    ]
    cells[6]["source"] = [
        line
        for line in """import torch
from google.colab import userdata
from huggingface_hub import snapshot_download
from peft import PeftModel
from transformers import (
    AutoModelForCausalLM,
    AutoModelForMultimodalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)

try:
    HF_TOKEN = userdata.get("HF_TOKEN")
except Exception:
    HF_TOKEN = os.environ.get("HF_TOKEN")

EXPECTED_BASE_MODEL_ID = os.environ.get(
    "FALSIFYRL_BASE_MODEL_ID",
    "Qwen/Qwen3.5-9B",
)
MODEL_REPO_ID = os.environ.get(
    "FALSIFYRL_MODEL_REPO_ID",
    "KuanKuanKuan/falsifyrl-autoscientist",
)
RUN_ID = os.environ.get(
    "FALSIFYRL_RUN_ID",
    "2f10c842-c124-407b-89c0-f4af5a761bb4",
)
STAGING_REPO_ID = os.environ.get(
    "FALSIFYRL_STAGING_REPO_ID",
    "__FALSIFYRL_STAGING_REPO_ID__",
)
STAGING_REVISION = os.environ.get(
    "FALSIFYRL_STAGING_REVISION",
    "__FALSIFYRL_STAGING_REVISION__",
)
STAGING_ADAPTER_PATH = os.environ.get(
    "FALSIFYRL_STAGING_ADAPTER_PATH",
    "__FALSIFYRL_STAGING_ADAPTER_PATH__",
).strip("/")
STAGING_MANIFEST_PATH = f"runs/{RUN_ID}/checkpoint-manifest.json"

def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

if STAGING_REPO_ID:
    assert HF_TOKEN, "enable HF_TOKEN for this private staging notebook"
    assert len(STAGING_REVISION) == 40, "pin the immutable 40-character staging commit"
    assert STAGING_ADAPTER_PATH, "private staging adapter path is required"
    staging_root = Path(snapshot_download(
        STAGING_REPO_ID,
        repo_type="model",
        revision=STAGING_REVISION,
        token=HF_TOKEN,
        allow_patterns=[f"{STAGING_ADAPTER_PATH}/*", STAGING_MANIFEST_PATH],
    ))
    ADAPTER_DIR = staging_root / STAGING_ADAPTER_PATH
    checkpoint_manifest = json.loads(
        (staging_root / STAGING_MANIFEST_PATH).read_text()
    )
    assert checkpoint_manifest["autoscientist_run_id"] == RUN_ID
    assert checkpoint_manifest["base_model_id"] == EXPECTED_BASE_MODEL_ID
    assert checkpoint_manifest["adapter_path"] == STAGING_ADAPTER_PATH
    assert checkpoint_manifest["adapted_dataset"]["test_jsonl_sha256"] == file_sha256(TEST_PATH)
    assert checkpoint_manifest["adapter_model"]["sha256"] == file_sha256(
        ADAPTER_DIR / "adapter_model.safetensors"
    )
    ADAPTER_SOURCE = f"{STAGING_REPO_ID}@{STAGING_REVISION}:{STAGING_ADAPTER_PATH}"
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
USE_4BIT = os.environ.get("FALSIFYRL_USE_4BIT", "false").lower() == "true"
if USE_4BIT:
    assert torch.cuda.is_available(), (
        "4-bit inference requires a Colab GPU runtime; select L4 or A100 before rerunning"
    )
    try:
        import bitsandbytes  # noqa: F401
    except ImportError as error:
        raise RuntimeError(
            "4-bit inference was requested but bitsandbytes is unavailable; rerun the install cell"
        ) from error
    model_kwargs["quantization_config"] = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=(
            torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        ),
    )
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

COLAB_OUTPUT_DIR = Path("/content/FalsifyRL/evaluation") / RUN_ID
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
artifact_manifest = {
    "schema_version": 1,
    "autoscientist_run_id": RUN_ID,
    "base_model_id": BASE_MODEL_ID,
    "checkpoint_revision": STAGING_REVISION or None,
    "adapter_sha256": report["adapter_sha256"],
    "test_jsonl_sha256": file_sha256(TEST_PATH),
    "example_count": MAX_EXAMPLES,
    "batch_size": BATCH_SIZE,
    "max_new_tokens": int(os.environ.get("FALSIFYRL_MAX_NEW_TOKENS", 768)),
    "do_sample": False,
    "files": {
        base_prediction_path.name: {
            "sha256": report["base_predictions_sha256"],
            "bytes": base_prediction_path.stat().st_size,
        },
        adapted_prediction_path.name: {
            "sha256": report["adapted_predictions_sha256"],
            "bytes": adapted_prediction_path.stat().st_size,
        },
        report_path.name: {
            "sha256": sha256(report_path),
            "bytes": report_path.stat().st_size,
        },
    },
}
artifact_manifest_path = COLAB_OUTPUT_DIR / "evaluation-manifest.json"
artifact_manifest_path.write_text(
    json.dumps(artifact_manifest, indent=2, sort_keys=True) + "\\n"
)
if STAGING_REPO_ID:
    from huggingface_hub import CommitOperationAdd, HfApi

    api = HfApi(token=HF_TOKEN)
    current_head = api.repo_info(
        repo_id=STAGING_REPO_ID,
        repo_type="model",
    ).sha
    evidence_path = f"runs/{RUN_ID}/evaluation"
    operations = [
        CommitOperationAdd(
            path_in_repo=f"{evidence_path}/{path.name}",
            path_or_fileobj=str(path),
        )
        for path in (
            base_prediction_path,
            adapted_prediction_path,
            report_path,
            artifact_manifest_path,
        )
    ]
    commit = api.create_commit(
        repo_id=STAGING_REPO_ID,
        repo_type="model",
        operations=operations,
        commit_message=f"Upload private Colab evaluation evidence for {RUN_ID}",
        parent_commit=current_head,
    )
    print("private evaluation revision:", commit.oid)
print(json.dumps(report, indent=2, sort_keys=True))
print("saved:", COLAB_OUTPUT_DIR)
""".splitlines(keepends=True)
    ]
    cells[10]["source"] = [
        line
        for line in """When private staging is configured, the notebook uploads both prediction
JSONL files and their hash manifest to the private repository. Otherwise download them from the
Colab file browser. The repository's CPU-only evaluator validates the complete output schema and
re-executes every proposed reward patch.
This Colab notebook performs only the GPU-heavy base and adapter inference over the exact same 640
held-out examples.
""".splitlines(keepends=True)
    ]
    replacements = {
        DEFAULT_BASE_MODEL_ID: base_model_id,
        DEFAULT_RUN_ID: run_id,
        DEFAULT_ARCHIVE_NAME: archive_name,
        STAGING_REPO_PLACEHOLDER: staging_repo_id or "",
        STAGING_REVISION_PLACEHOLDER: staging_revision or "",
        STAGING_ADAPTER_PATH_PLACEHOLDER: staging_adapter_path or "",
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
    parser.add_argument("--staging-repo-id")
    parser.add_argument("--staging-revision")
    parser.add_argument("--staging-adapter-path")
    parser.add_argument("--max-examples", type=int)
    parser.add_argument("--max-new-tokens", type=int, default=768)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument(
        "--use-4bit",
        action="store_true",
        help="Install bitsandbytes and load the base model with 4-bit NF4 quantization.",
    )
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
                staging_repo_id=args.staging_repo_id,
                staging_revision=args.staging_revision,
                staging_adapter_path=args.staging_adapter_path,
                max_examples=args.max_examples,
                max_new_tokens=args.max_new_tokens,
                batch_size=args.batch_size,
                use_4bit=args.use_4bit,
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
