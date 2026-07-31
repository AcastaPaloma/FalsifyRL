# FalsifyRL

FalsifyRL is an evidence-grounded critic for reward hacking in embodied, multi-agent reinforcement
learning. Given a task specification, a declarative proxy reward, and an episode trace, it returns a
strict JSON diagnosis with evidence, responsible agents, a counterexample, and an executable reward
patch.

The project is being built as a standalone Science-track entry for the Adaption AutoScientist
Challenge. Gold labels come from deterministic generators and independent task validators—not from
language-model annotation.

## Why it matters

An RL policy can maximize a proxy reward while violating the task that reward was meant to encode:
one robot can free-ride on a teammate, unsafe motion can be ignored, or a survival bonus can make
doing nothing optimal. FalsifyRL turns those failures into a reproducible falsification benchmark
and trains a compact model to recognize and repair them.

## Current status

The deterministic core supports four disjoint scenario families and eight reward-defect classes.
The v1 seed generator creates reward-matched control/exploit pairs, verifies every example with the
independent task oracle, and exports JSONL plus two-column AutoScientist CSV files. Baseline
evaluation, AutoScientist training, and public Hugging Face/Kaggle release tooling are the next
milestones.

See [docs/hackathon_plan.md](docs/hackathon_plan.md) for the complete submission contract.
The credential-safe platform sequence is in
[docs/autoscientist_runbook.md](docs/autoscientist_runbook.md).

## Development

```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[dev,platforms]"
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m ruff check .
.\.venv\Scripts\python.exe scripts/generate_seed_dataset.py
.\.venv\Scripts\python.exe scripts/evaluate_baselines.py --baseline reward-only --split test
.\.venv\Scripts\python.exe scripts/autoscientist_workflow.py plan --source huggingface `
  --source-url https://huggingface.co/datasets/OWNER/falsifyrl-seed
```

The held-out reward-only baseline gets 50% verdict accuracy and 0.333 verdict macro-F1 because each
reward program is shared by one control and one exploit. The executable oracle ceiling is 1.0 on
every metric.
