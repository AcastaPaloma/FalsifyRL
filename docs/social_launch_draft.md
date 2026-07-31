# FalsifyRL Launch Drafts

Do not publish these drafts until the final submission audit passes. Replace every `{{...}}`
placeholder with a value copied from `outputs/submission/manifest.json`, then tag the official
Adaption account in the posting UI.

## LinkedIn

What if an RL agent maximizes its reward by doing the wrong thing—and the evaluator can prove it?

For the Adaption Labs AutoScientist Challenge, I built **FalsifyRL**: an evidence-grounded critic
for reward hacking in embodied, multi-agent reinforcement learning.

Each example contains a task specification, a declarative proxy reward, and an episode trace. The
critic returns a strict JSON diagnosis with responsible agents, evidence steps, a counterexample,
and an executable reward patch.

The experiment is designed to be falsifiable:

- 3,840 deterministic examples in 1,920 reward-matched control/exploit pairs
- train, validation, and test scenario families are disjoint
- gold labels come from executable validators rather than language-model annotation
- predicted reward patches are replayed instead of graded by text similarity
- the final model is evaluated on all 640 held-out `crossing_navigation` cases

Using Adaptive Data and AutoScientist, the adapted model reached an AutoScientist best win rate of
**{{AUTOSCIENTIST_BEST_WIN_RATE}}** and improved held-out composite score from **0.0000** to
**{{TRAINED_COMPOSITE_SCORE}}**, with **{{TRAINED_JSON_VALIDITY}}** JSON validity.

Everything is open and reproducible:

- Code: {{GITHUB_URL}}
- Dataset: https://huggingface.co/datasets/KuanKuanKuan/falsifyrl-adapted
- Model: {{HUGGINGFACE_MODEL_URL}}
- Live demo: {{HUGGINGFACE_SPACE_URL}}
- Reproducible evaluation: {{KAGGLE_NOTEBOOK_URL}}
- Evaluation report: {{EVALUATION_REPORT_URL}}

Falsify reward functions before your robots exploit them.

#AutoScientistChallenge #AdaptionLabs #ReinforcementLearning #MultiAgentSystems #Robotics
#PhysicalAI #AISafety #OpenSourceAI

## X thread

Post 1:

> I built FalsifyRL for the @{{ADAPTION_X_HANDLE}} AutoScientist Challenge: a critic that finds
> reward exploits in embodied multi-agent RL, identifies responsible agents, and emits executable
> repairs.
>
> 3,840 verified examples + a family-disjoint 640-case test.
>
> #AutoScientistChallenge

Post 2:

> Held-out composite: 0.0000 → {{TRAINED_COMPOSITE_SCORE}}
> AutoScientist win rate: {{AUTOSCIENTIST_BEST_WIN_RATE}}
>
> Demo: {{HUGGINGFACE_SPACE_URL}}
> Code + reproducibility: {{GITHUB_URL}}
>
> #ReinforcementLearning #PhysicalAI

## Pre-publish checks

- No `{{...}}` placeholders remain.
- Every metric exactly matches the anonymous evaluation report.
- Every URL opens in a logged-out browser.
- The official Adaption account is tagged, not merely written as plain text.
- The entrant approves the final wording and explicitly authorizes each post.
