# Final AutoScientist Challenge Submission Checklist

The command below is the final fail-closed gate:

```powershell
.\.venv\Scripts\python.exe scripts/audit_submission.py
```

It must exit zero before the official form is submitted.

## Entrant attestations

The user must personally confirm:

- accepted into the AutoScientist Challenge,
- at least 18 or the age of majority,
- not a resident of Québec,
- participation is legal in their jurisdiction,
- membership in only one challenge team.

These are personal/legal facts and are never inferred by the implementation.

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

Copy values only from the audited `outputs/submission/manifest.json`. Save the form confirmation
page or email under the ignored `outputs/submission/` directory.

