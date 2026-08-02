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

## Finalize and publish the trained model

The private staging repository uses two different immutable revisions:

- the checkpoint revision recorded when the exact AutoScientist adapter was staged, and
- the later evidence revision created after Colab uploads all 640 held-out predictions.

Never substitute one for the other. Finalize CPU-side metrics with both revisions and run-scoped
paths:

```powershell
$run = "255e1c38-a488-45ea-ac90-21e579d6c119"
$checkpointRevision = "1db2801bdf26d793eb24fe6071a4f46018a49047"
$evidenceRevision = "COLAB_EVIDENCE_COMMIT"

.\.venv\Scripts\python.exe scripts/finalize_external_evaluation.py `
  --state "outputs/evaluation/$run/workflow.json" `
  --staging-repo-id KuanKuanKuan/falsifyrl-eval-staging `
  --evidence-revision $evidenceRevision `
  --checkpoint-revision $checkpointRevision `
  --adapter-weights "outputs/evaluation/$run/adapter_model.safetensors" `
  --dataset-manifest artifacts/release/adapted-dataset/release-manifest.json `
  --output-dir "outputs/evaluation/$run/final" `
  --submission-manifest outputs/submission/manifest.json
```

The final model bundle contains at least:

```text
README.md
adapter_config.json
adapter_model.safetensors
tokenizer files
evaluation report
LICENSE
```

The model card must replace all run-specific placeholders before either publisher will proceed.
For the selected Llama checkpoint, use the genuine Llama 3.2 Community License and leave Kaggle's
license metadata unset rather than claiming Apache 2.0. The bundled license remains authoritative.

```powershell
.\.venv\Scripts\python.exe -m scripts.continue_model_release `
  --state "outputs/evaluation/$run/workflow.json" `
  --checkpoint outputs/autoscientist/best-checkpoint.tgz `
  --adapter-dir "outputs/evaluation/$run/release-adapter" `
  --checkpoint-manifest "outputs/evaluation/$run/checkpoint-manifest.json" `
  --comparison "outputs/evaluation/$run/final/comparison.json" `
  --base-predictions "outputs/evaluation/$run/final/staged-evidence/falsifyrl-base-test-predictions.jsonl" `
  --evaluation-evidence-dir "outputs/evaluation/$run/final/staged-evidence" `
  --model-predictions "outputs/evaluation/$run/final/staged-evidence/falsifyrl-adapted-test-predictions.jsonl" `
  --model-bundle "artifacts/release/$run/model" `
  --space-bundle "artifacts/release/$run/space" `
  --test-jsonl outputs/falsifyrl_seed_v1/test.jsonl `
  --huggingface-owner KuanKuanKuan `
  --kaggle-owner kuanyiwang `
  --model-slug Llama-FalsifyRL-AutoScientist `
  --kaggle-model-version 1 `
  --selected-release-record "outputs/evaluation/$run/selected-release.json" `
  --space-slug falsifyrl-llama `
  --model-card-template release/model/README.llama3.2.md `
  --model-license-file outputs/licenses/Llama-3.2-LICENSE.txt `
  --kaggle-license-name ""
```

The script re-extracts the chosen checkpoint into the empty run-scoped adapter directory, rejects a
base-model mismatch, publishes the exact same adapter bytes to both hosts, and verifies both public
hashes. The resulting Kaggle model handle is
`kuanyiwang/Llama-FalsifyRL-AutoScientist/pytorch/lora`.

The release transaction stages the Hugging Face model privately. KaggleHub creates new models
private, and the release verifies both adapter hashes with authenticated clients before promoting
either model. Kaggle's upload API does not expose a visibility argument, so the publisher also
requests the intended visibility explicitly. Any later failure attempts to restore the Hugging Face
model/Space and Kaggle model to private visibility; inspect the raised rollback details before
retrying.

If Kaggle's authenticated `UpdateModel` endpoint returns HTTP 403 after the exact model
version has uploaded, keep both model repositories private and make that existing Kaggle model
public from its Settings page. Do not rerun the uploader, because that would create a second
version. Resume without re-uploading by running `scripts/resume_model_release.py` against the
populated run-scoped model and Space bundles. The resume path verifies both remote adapter hashes
and the anonymous Kaggle page before promoting either Hugging Face artifact, and it never calls
Kaggle's rejected visibility endpoint.

For the selected run, after making **version 1 of the existing model** public in Kaggle Settings:

```powershell
$run = "255e1c38-a488-45ea-ac90-21e579d6c119"

.\.venv\Scripts\python.exe -m scripts.resume_model_release `
  --state "outputs/evaluation/$run/workflow.json" `
  --checkpoint outputs/autoscientist/best-checkpoint.tgz `
  --adapter-dir "outputs/evaluation/$run/release-adapter-resume-20260802b" `
  --checkpoint-manifest "outputs/evaluation/$run/checkpoint-manifest.json" `
  --comparison "outputs/evaluation/$run/final/comparison.json" `
  --submission-manifest outputs/submission/manifest.json `
  --model-bundle "artifacts/release/$run/model-rerun-20260802" `
  --space-bundle "artifacts/release/$run/space-static-rerun-20260802" `
  --base-predictions "outputs/evaluation/$run/final/staged-evidence/falsifyrl-base-test-predictions.jsonl" `
  --model-predictions "outputs/evaluation/$run/final/staged-evidence/falsifyrl-adapted-test-predictions.jsonl" `
  --huggingface-owner KuanKuanKuan `
  --kaggle-owner kuanyiwang `
  --model-slug Llama-FalsifyRL-AutoScientist `
  --kaggle-model-version 1 `
  --selected-release-record "outputs/evaluation/$run/selected-release.json" `
  --space-slug falsifyrl-llama
```

The resume adapter directory must be new and empty. The script refuses an existing selected-release
record and never calls either model uploader, so it cannot create Kaggle version 2.

## Publish the interactive Space

Prepare the Space bundle from held-out examples:

```powershell
.\.venv\Scripts\python.exe -m scripts.prepare_space `
  --prediction-jsonl "outputs/evaluation/$run/final/staged-evidence/falsifyrl-adapted-test-predictions.jsonl"
```

The release command above prepares and publishes the cached-prediction Space only after both model
repositories verify. Verify it separately:

```powershell
.\.venv\Scripts\python.exe -m scripts.continue_space_verification `
  --submission-manifest outputs/submission/manifest.json `
  --examples "artifacts/release/$run/space-static-rerun-20260802/examples.json" `
  --output "outputs/evaluation/$run/space-verification.json"
```

The publisher uploads a free static evidence explorer with 16 examples representing eight
reward-matched control/exploit pairs. It serves the strict cached outputs from the exact checkpoint
and their source prediction-file hash; the verifier anonymously checks the page, bundle, and one
control/exploit diagnosis after the Space finishes building.

## Publish the held-out Kaggle notebook

After the adapted dataset and model exist on Kaggle, stage the notebook with both resources
declared as immutable inputs. The public CPU run recomputes all compact metrics from the complete
Colab prediction evidence bundled with the hash-verified adapter, so it does not need gated base
weights or a secret. A copied GPU notebook can optionally set `FALSIFYRL_LIVE_INFERENCE=1` and use
a private `HF_TOKEN` Kaggle secret to regenerate those predictions.

```powershell
.\.venv\Scripts\python.exe -m scripts.continue_kaggle_notebook `
  --selected-release-record "outputs/evaluation/$run/selected-release.json" `
  --owner kuanyiwang `
  --model-slug Llama-FalsifyRL-AutoScientist `
  --model-version 1 `
  --submission-manifest outputs/submission/manifest.json `
  --bundle-dir "artifacts/release/$run/kaggle-notebook" `
  --output-dir "outputs/evaluation/$run/kaggle-notebook"
```

If the model is not version 1, pass its actual public version. Do not remove `model_sources` or
substitute the source dataset: the notebook must run against the exact `falsifyrl-adapted` release
and exact selected `Llama-FalsifyRL-AutoScientist/pytorch/lora/<version>` adapter. The continuation
script waits for completion, downloads the output report, audits it, and records the public notebook
URL in the private submission manifest.
