# AutoScientist Runbook

This runbook is intentionally staged. Each stage writes only public IDs and status fields to the
ignored workflow state. Credentials remain in environment variables.

Official references:

- [AutoScientist Python API guide](https://docs.adaptionlabs.ai/guides/autoscientist-api/)
- [Create an AutoScientist run](https://docs.adaptionlabs.ai/api/resources/autoscientist/methods/create)
- [Create/import a dataset](https://docs.adaptionlabs.ai/api/resources/datasets/methods/create)
- [Start or estimate an Adaptive Data run](https://docs.adaptionlabs.ai/api/resources/datasets/methods/run)
- [List supported training models](https://docs.adaptionlabs.ai/api/resources/training_models/methods/list)

The official dataset publish endpoint currently documents a `501 Not Implemented` response.
FalsifyRL therefore publishes to Hugging Face and Kaggle with their own clients, then imports the
public Hugging Face dataset into Adaption.

## 1. Create the local workflow plan

This is offline and does not require credentials:

```powershell
python scripts/autoscientist_workflow.py plan `
  --source huggingface `
  --source-url https://huggingface.co/datasets/OWNER/falsifyrl-seed
```

The committed default is three AutoScientist iterations with a target win rate of 0.75. The model
is left unset so the platform can choose based on dataset size.

## 2. Ingest and estimate

After the public dataset exists, use the locally hash-matched JSONL copy for ingestion. Adaption's
Hugging Face CSV importer counted multiline prompt rows incorrectly during the live run (2,595
instead of 2,560); the JSONL upload preserves the same verified prompt/completion records and
imports exactly 2,560 rows.

```powershell
python scripts/autoscientist_workflow.py plan `
  --source file `
  --local-file outputs/falsifyrl_seed_v1/train.jsonl
python scripts/autoscientist_workflow.py ingest
```

The workflow treats Adaption's `awaiting_input` status as a completed ingestion ready for column
mapping, persists the dataset ID immediately, requires exactly 2,560 rows, and calls
`datasets.run(..., estimate=True)`. It does not launch a paid adaptation. Inspect
`estimated_credits` and `estimated_minutes` in `outputs/autoscientist/workflow.json`.

## 3. Adapt the dataset

```powershell
python scripts/autoscientist_workflow.py adapt
```

Prompt rephrasing, reasoning-trace generation, and deduplication are disabled because FalsifyRL uses
reward-matched pairs and strict simulator-derived JSON labels. The blueprint asks Adaptive Data to
preserve the evidence and output schema.

Export and revalidate the exact adapted rows before training:

```powershell
python scripts/autoscientist_workflow.py export
```

The export gate selects `enhanced_prompt` / `enhanced_completion` when present, otherwise the
original `prompt` / `completion` fields. It then requires all 2,560 source prompts exactly once,
strict JSON completions, and preservation of the simulator-derived verdict, failure type,
responsible agents, evidence, counterexample, and executable patch. It records the export and audit
hashes in workflow state. AutoScientist training is blocked until this audit succeeds.

## 4. Train

```powershell
python scripts/autoscientist_workflow.py train
```

The request uses instruction format, an idempotency key derived from the dataset ID, and platform
defaults for the base model, LoRA recipe, and hyperparameters. Adaption SDK 0.6.2 exposes the
training strategy under `hyperparams.training_type`; the official default is LoRA, so the adapter
does not override it.

The resulting dataset ID, experiment/model ID, base model, best win rate, and status are recorded in
the ignored workflow state for the final submission manifest.

## 5. Download the best checkpoint

```powershell
python scripts/autoscientist_workflow.py status
python scripts/autoscientist_workflow.py download
```

The download is the best iteration checkpoint, not necessarily the final iteration. Checkpoints are
ignored by Git and later uploaded to Hugging Face Models and Kaggle Models with cards that identify
the base model and license.
