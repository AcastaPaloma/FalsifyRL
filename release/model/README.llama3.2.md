---
license: llama3.2
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

# Llama FalsifyRL AutoScientist Critic

This repository contains the best LoRA checkpoint produced by Adaption AutoScientist for
FalsifyRL, an evidence-grounded critic for reward hacking in embodied multi-agent reinforcement
learning.

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

- AutoScientist experiment: `AUTOSCIENTIST_RUN_ID`
- base model: `BASE_MODEL_ID`
- best platform win rate: `BEST_WIN_RATE`
- held-out metrics: `EVALUATION_REPORT_URL`

## License and attribution

This adapter is derived from Meta Llama 3.2 and is distributed subject to the Llama 3.2 Community
License included in this repository. The repository and model name begin with “Llama” to preserve
the required naming attribution. The FalsifyRL source code and generated dataset remain MIT
licensed.

## Safety and limitations

This model proposes simulator-checkable reward patches; it is not an autonomous robot-safety
system. Always run the executable verifier and perform expert review before changing a deployed
reward function.
