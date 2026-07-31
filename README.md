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

## Development

```powershell
python -m pip install -e .[dev]
python -m pytest -q
python -m ruff check .
python scripts/generate_seed_dataset.py
```
