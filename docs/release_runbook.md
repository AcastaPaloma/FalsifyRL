# Hugging Face and Kaggle Release Runbook

FalsifyRL uses native Hugging Face and Kaggle clients because Adaption's dataset publish endpoint is
currently documented as unimplemented.

Official references:

- [Hugging Face dataset cards](https://huggingface.co/docs/hub/datasets-cards)
- [Hugging Face folder upload](https://huggingface.co/docs/huggingface_hub/guides/upload)
- [KaggleHub dataset and model upload](https://github.com/Kaggle/kagglehub)

## Publish the verified source dataset for Adaption import

```powershell
.\.venv\Scripts\python.exe scripts/prepare_release.py
```

This rechecks every generated SHA-256 digest and stages the verified source data under
`artifacts/release/dataset/`. Publish this initial bundle so Adaption can import `train.csv`.

## Credentials

Use process environment variables only:

```powershell
$env:HF_TOKEN = "..."
$env:KAGGLE_API_TOKEN = "..."
```

The Hugging Face token needs write access. Generate the Kaggle token from the current API-token
settings. Alternatively, copy `.env.example` to the ignored `.env` file and fill the token and
owner fields there. Do not pass either token on the command line.

```powershell
.\.venv\Scripts\python.exe scripts/publish_dataset.py huggingface `
  --owner OWNER --slug falsifyrl-source
```

After Adaptive Data succeeds, run the mandatory export/audit step from the AutoScientist runbook,
then prepare the exact adapted release:

```powershell
.\.venv\Scripts\python.exe scripts/prepare_adapted_release.py
```

This stages the audited Adaptive Data export as `train.csv`, the original source rows as
`source_train.csv`, held-out validation/test data, both audit manifests, the card, and license under
`artifacts/release/adapted-dataset/`. The `train.csv` hash must match the export audited before
AutoScientist training.

## Publish the exact training dataset

```powershell
.\.venv\Scripts\python.exe scripts/publish_dataset.py huggingface `
  --owner OWNER --slug falsifyrl-adapted `
  --bundle-dir artifacts/release/adapted-dataset
.\.venv\Scripts\python.exe scripts/publish_dataset.py kaggle `
  --owner OWNER --slug falsifyrl-adapted `
  --bundle-dir artifacts/release/adapted-dataset
```

Both repositories must be public. After publication, verify all links in a logged-out browser,
confirm the Hugging Face `autoscientist` config resolves all three splits, and verify the public
`train.csv` hash matches the workflow state's audited export hash. Use these adapted-dataset URLs in
the challenge submission.

## Publish the trained model

After the best AutoScientist checkpoint is downloaded, prepare a model bundle containing at least:

```text
README.md
adapter_config.json
adapter_model.safetensors
tokenizer files
evaluation report
LICENSE
```

The model card must replace all run-specific placeholders before either publisher will proceed.

```powershell
.\.venv\Scripts\python.exe scripts/publish_model.py huggingface --owner OWNER
.\.venv\Scripts\python.exe scripts/publish_model.py kaggle --owner OWNER
```

The Kaggle model handle is published as
`OWNER/falsifyrl-autoscientist/pytorch/lora`.

## Publish the interactive Space

Prepare the Space bundle from held-out examples:

```powershell
.\.venv\Scripts\python.exe scripts/prepare_space.py
```

Publish only after the model repository exists:

```powershell
.\.venv\Scripts\python.exe scripts/publish_space.py `
  --owner OWNER `
  --base-model-id BASE_MODEL_ID `
  --model-repo-id OWNER/falsifyrl-autoscientist
```

The publisher uploads 16 examples representing eight reward-matched control/exploit pairs and sets
the public Space variables needed for lazy model loading. Verify both a control and exploit
prediction after the Space finishes building.
