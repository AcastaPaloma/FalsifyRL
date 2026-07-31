---
license: mit
library_name: peft
tags:
  - reinforcement-learning
  - multi-agent
  - robotics
  - reward-hacking
  - autoscientist
base_model: BASE_MODEL_ID
datasets:
  - DATASET_REPO_ID
---

# FalsifyRL AutoScientist Critic

This repository contains the best LoRA checkpoint produced by Adaption AutoScientist for FalsifyRL,
an evidence-grounded critic for reward hacking in embodied multi-agent reinforcement learning.

## Input

A task specification, declarative proxy reward, and compact episode trace.

## Output

Exactly one JSON object with:

```text
verdict
failure_type
responsible_agents
evidence_steps
counterexample_config
reward_patch
expected_effect
confidence
```

## Evaluation

Final held-out metrics and the AutoScientist experiment ID will be inserted after training:

- AutoScientist experiment: `AUTOSCIENTIST_RUN_ID`
- base model: `BASE_MODEL_ID`
- best platform win rate: `BEST_WIN_RATE`
- held-out metrics: `EVALUATION_REPORT_URL`

## Safety and limitations

This model proposes simulator-checkable reward patches; it is not an autonomous robot-safety
system. Always run the executable verifier and perform expert review before changing a deployed
reward function.

