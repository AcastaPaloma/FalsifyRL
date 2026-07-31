# Hugging Face and Kaggle Release Runbook

FalsifyRL uses native Hugging Face and Kaggle clients because Adaption's dataset publish endpoint is
currently documented as unimplemented.

Official references:

- [Hugging Face dataset cards](https://huggingface.co/docs/hub/datasets-cards)
- [Hugging Face folder upload](https://huggingface.co/docs/huggingface_hub/guides/upload)
- [KaggleHub dataset and model upload](https://github.com/Kaggle/kagglehub)

## Prepare and audit the dataset bundle

```powershell
.\.venv\Scripts\python.exe scripts/prepare_release.py
```

This rechecks every generated SHA-256 digest, stages the six data files, source manifest, dataset
card, license, and a release manifest under `artifacts/release/dataset/`.

## Credentials

Use process environment variables only:

```powershell
$env:HF_TOKEN = "..."
$env:KAGGLE_API_TOKEN = "..."
```

The Hugging Face token needs write access. Generate the Kaggle token from the current API-token
settings. Do not pass either token on the command line.

## Publish the dataset

```powershell
.\.venv\Scripts\python.exe scripts/publish_dataset.py huggingface --owner OWNER
.\.venv\Scripts\python.exe scripts/publish_dataset.py kaggle --owner OWNER
```

Both repositories must be public. After publication, verify all links in a logged-out browser,
confirm the Hugging Face dataset viewer resolves the three splits, and import the Hugging Face
`train.csv` into Adaption.

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

