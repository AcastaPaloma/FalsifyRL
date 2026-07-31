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
- `src/falsifyrl/dataset.py`: paired-case assembly, validation, and deterministic export.
- `scripts/generate_seed_dataset.py`: seed dataset CLI.
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

### 2026-07-30 Deterministic Core

- Implemented strict schemas, four disjoint scenario families, eight reward-defect generators plus
  aligned controls, declarative reward patches, and an independent task oracle.
- The verifier requires a failed true-task outcome, proxy return at least 90% of the aligned return,
  an exploit reduction of at least 0.5 after patching, positive aligned return, and a patched
  aligned-over-exploit margin of at least 0.5.
- Extended the no-op exploit horizon so it is genuinely reward-competitive instead of merely
  invalid behavior.
- Added schema, deterministic-generation, split-isolation, leakage-marker, strict-JSON, and
  executable-verifier tests.
- Validation: `python -m pytest -q` -> 9 passed; `python -m ruff check .` -> passed; 720 generated
  cases across 20 seeds passed executable verification.

### 2026-07-30 Verified Seed Dataset

- Added reward-matched aligned/exploit pairs: both members expose the same reward program, which
  forces the model to reason over the episode trace instead of classifying reward code alone.
- Added deterministic JSONL and AutoScientist-ready CSV export with exact hashes and a validation
  manifest.
- Default v1 release candidate contains 3,840 cases / 1,920 pairs: 2,560 train, 640 validation, and
  640 held-out test examples. Verdicts are exactly balanced.
- The generated local release candidate is under `outputs/falsifyrl_seed_v1/` and is ignored by Git.
- Validation: `python -m pytest -q` -> 12 passed; `python -m ruff check .` -> passed; full generation
  reported all 3,840 cases verified and all pairs reward-matched.

### 2026-07-30 Evaluation Harness

- Added strict prediction JSONL loading and metrics for JSON validity, verdict and failure-type
  accuracy/macro-F1, responsible-agent exact match, evidence-step F1, executable patch success, and
  a five-part composite.
- Added always-aligned, reward-program-only, and executable-oracle baselines.
- On the 640-case held-out `crossing_navigation` split, both non-reasoning baselines have 0.500
  verdict accuracy and 0.333 verdict macro-F1; the reward-only baseline composite is 0.485. The
  executable oracle scores 1.0.
- Validation: `python -m pytest -q` -> 16 passed; `python -m ruff check .` -> passed.
