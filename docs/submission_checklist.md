# Final AutoScientist Challenge Submission Checklist

The command below is the final fail-closed gate:

```powershell
.\.venv\Scripts\python.exe scripts/audit_submission.py
```

It must exit zero before the official form is submitted.

Part 2 closes **August 10, 2026**. The live form was inspected on July 31, 2026.
Submit under **Science**.

## Entrant attestations

The user must personally confirm:

- accepted into the AutoScientist Challenge,
- at least 18,
- not a resident of Québec,
- participation is legal in their jurisdiction,
- membership in only one challenge team,
- agreement to the linked contest Terms and Conditions.

These are personal/legal facts and are never inferred by the implementation.

## Private form inputs

Populate these only in the ignored `outputs/submission/manifest.json`, never in the public
template, repository, terminal output, or model/data cards:

- first name, last name, email,
- optional phone number,
- job title and company name,
- street address, city, state/region, postal code, country,
- optional team-member and team-captain details,
- Discord username,
- an explicit yes/no answer for HackIndia participation.

The live form also asks for the Adaption dataset ID, optional Training Model ID, final dataset,
model weights, a completed Kaggle URL, a completed Hugging Face URL, and optional LinkedIn/X
post links. Map those only from the audited identifiers and public links below.

## Required artifacts

- public GitHub repository with clean structured history,
- exact trainable dataset on Hugging Face Datasets,
- same dataset version on Kaggle Datasets,
- Adaption dataset ID,
- AutoScientist experiment/run ID,
- public best LoRA checkpoint on Hugging Face Models,
- same checkpoint on Kaggle Models,
- public Hugging Face Space using the trained checkpoint,
- public Kaggle held-out evaluation notebook,
- anonymous-access evaluation report,
- positive held-out improvement over the unadapted base model,
- AutoScientist best win rate above 0.5,
- trained JSON validity of at least 95%.

## Reproducibility evidence

- generator version and configuration,
- six source-data SHA-256 hashes,
- family-disjoint split manifest,
- executable verifier pass count,
- reward-matched pair count,
- base-model prediction JSONL and metric report,
- trained-model prediction JSONL and metric report,
- best checkpoint release manifest,
- exact base model ID and license,
- Kaggle notebook output report.

## Final link audit

Open every required link in a logged-out browser. Confirm:

- no login is required,
- Hugging Face displays all three dataset splits,
- Kaggle dataset files match the release manifest,
- both model hosts contain identical adapter hashes,
- Space predicts both a control and exploit trace,
- Kaggle notebook has a successful public run,
- evaluation report contains the held-out `crossing_navigation` metrics.

## Submission

Official Part 2 form:

<https://share.hsforms.com/2xleXmJ7wSkimSzP8L55KcAuc9yb>

Recommended field mapping:

- **Specify dataset ID:** `identifiers.adaption_dataset_id`
- **Training Model ID:** `identifiers.autoscientist_run_id`
- **Insert final dataset:** `links.huggingface_dataset` and `links.kaggle_dataset`
- **Insert model weights link:** `links.huggingface_model` and `links.kaggle_model`
- **Completed open source Kaggle URL:** include the Kaggle dataset, model, and notebook URLs
- **Completed open source Hugging Face URL:** include the dataset, model, and Space URLs
- **LinkedIn and X posts:** the two audited social links

Copy values only from the audited `outputs/submission/manifest.json`. Save the form confirmation
page or email under the ignored `outputs/submission/` directory.

After all public links have passed the audit, render the approval-ready social drafts without
copying metrics or URLs by hand:

```powershell
.\.venv\Scripts\python.exe scripts/render_social_launch.py
```

This writes ignored `outputs/submission/social_launch_ready.md`; it does not post anything.
