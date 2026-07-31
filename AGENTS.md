# AGENTS.md

This file is the persistent working context for FalsifyRL. Read it before non-trivial work and
update it whenever implementation direction, setup, validation, or release state changes.

## Project Goal

FalsifyRL is a standalone AutoScientist Challenge submission in the Science category. It trains a
small critic to detect and repair proxy-reward failures in embodied, multi-agent reinforcement
learning from a task specification, declarative reward program, and episode trace.

PaintMerge is not part of this repository and is not a runtime dependency. Its two-arm MuJoCo
environment may later provide an external visual evaluation case.

## Non-Negotiable Submission Gates

- Keep the generated labels deterministic and derived from executable validators, not an LLM.
- Split by scenario family so no family occurs in multiple dataset splits.
- Every reward-hack case must have an executable patch that suppresses the exploit while preserving
  aligned behavior.
- Never put defect names, validator outcomes, secrets, or hidden generation metadata in prompts.
- Publish the exact adapted dataset and trained weights on both Hugging Face and Kaggle.
- Demonstrate measurable held-out improvement over the selected base model.
- Keep AutoScientist, Hugging Face, and Kaggle credentials in environment variables or ignored
  `.env` files; never print or commit them.
- Use clean, scoped commits and do not commit generated weights or large run artifacts.

## Local Setup

The current validated interpreter is:

```powershell
C:\Users\Win10\anaconda3\envs\paintmerge-gtx1080\python.exe
```

The environment name is historical; FalsifyRL does not depend on PaintMerge.

```powershell
python -m pip install -e .[dev]
python -m pytest -q
python -m ruff check .
```

## Repository Map

- `docs/hackathon_plan.md`: challenge contract, scientific design, and release gates.
- `src/falsifyrl/schema.py`: strict reward, trace, patch, and diagnosis schemas.
- `src/falsifyrl/scenarios.py`: deterministic scenario-family and defect generators.
- `src/falsifyrl/verifier.py`: independent task oracle and executable patch checks.
- `tests/`: deterministic, leakage, and verifier regression tests.

## Context Log

### 2026-07-30 Standalone Repository

- Created FalsifyRL as `A:\projects\FalsifyRL` after the user clarified that the hackathon project
  must live outside PaintMerge.
- Kept PaintMerge as an optional future external evaluation source only.
- Selected the Science-track concept: falsify and repair reward specifications for embodied
  multi-agent RL.
- Challenge research, requirements, dataset/model publication gates, and credential policy are in
  `docs/hackathon_plan.md`.

