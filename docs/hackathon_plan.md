# FalsifyRL AutoScientist Challenge Plan

Date: 2026-07-30

## Submission Objective

Build and submit **FalsifyRL**, an AutoScientist-trained critic for falsifying reward functions in
embodied and multi-agent reinforcement learning.

The model receives:

- a natural-language task specification,
- a declarative reward program rendered as Python-like code,
- a compact, step-indexed episode trace.

It returns one strict JSON diagnosis containing:

- whether the episode falsifies the intended reward,
- the failure category,
- responsible agents,
- evidence steps,
- an executable counterexample configuration,
- a machine-applicable reward patch,
- the expected effect and calibrated confidence.

The Science-track claim is not that a language model directly controls a robot. The claim is that a
small adapted model can perform a useful part of the scientific method around embodied RL: identify
a mismatch between proxy reward and task intent, construct a counterexample, and propose a patch
that survives executable regression tests.

## Challenge Contract

Current Part 2 deadline: **August 10, 2026**.

Submission category: **Science**.

Required public artifacts:

1. Adapted fine-tuning dataset on Hugging Face.
2. Adapted fine-tuning dataset on Kaggle.
3. AutoScientist-trained model or LoRA weights on Hugging Face.
4. The same trained weights on Kaggle Models.
5. Measurable improvement over the base model.
6. Adaption dataset ID and training model ID.
7. Public demo, preferably a Hugging Face Space.
8. Reproducible Kaggle notebook.
9. Completed Part 2 submission form.

Bonus artifacts:

- LinkedIn and X posts tagging Adaption.
- Short comparison video or GIF.
- Clear model card, dataset card, limitations, and reproducibility instructions.

Eligibility gates that require the entrant to confirm:

- accepted into the challenge,
- at least 18 or age of majority,
- not a resident of Québec,
- participation is legal in the entrant's jurisdiction,
- each person belongs to only one team.

## Repository Boundary

FalsifyRL is a standalone repository:

```text
src/falsifyrl/
scripts/generate_falsifyrl_seed.py
scripts/evaluate_falsifyrl.py
tests/test_falsifyrl_*.py
```

FalsifyRL owns its generators, validators, benchmark splits, evaluation, and release pipeline. It
does not import another project or depend on a task-specific simulator. The held-out benchmark
therefore measures cross-task reward-falsification behavior rather than memorization of one
application's semantics.

## Scientific Design

Every generated case has two separate definitions:

```text
proxy reward: what an RL policy is told to maximize
true validator: executable task success and safety requirements
```

A case is a verified reward exploit when:

```text
proxy return is competitive with or exceeds the aligned trajectory
AND
the independent true validator rejects the exploit trajectory
```

Each injected reward defect also has:

- an aligned reference trajectory,
- an exploit trajectory,
- a known minimal patch,
- a verifier that checks the patch reduces exploit return,
- a preservation check that the patch does not destroy aligned behavior.

No language model is used to create gold labels. Labels are derived from the simulator and mutation
metadata so they are correct by construction.

## Initial Scenario Families

| Family | Embodied coordination problem | Split |
| --- | --- | --- |
| `dual_arm_workspace` | Two arms share a work surface and must both contribute safely | train |
| `warehouse_handoff` | Two robots divide and hand off object-moving work | train |
| `cooperative_transport` | Two agents jointly transport one payload | validation |
| `crossing_navigation` | Two agents cross a constrained shared workspace | test |

The family-level split prevents near-duplicate trajectories from appearing across train and test.

## Reward Defect Taxonomy

1. `aligned`: reward agrees with the true validator.
2. `free_riding`: max/shared progress rewards an inactive agent.
3. `collision_blind`: task progress ignores collisions.
4. `destructive_interference`: progress ignores undoing another agent's work.
5. `unsafe_speed`: completion ignores actuator or speed-limit violations.
6. `out_of_bounds`: progress ignores leaving the safe workspace.
7. `no_op_bonus`: a positive survival/step bonus outcompetes task completion.
8. `premature_termination`: partial progress triggers the completion bonus.
9. `credit_misattribution`: one agent's contribution is credited to the full team.

## Dataset Interface

AutoScientist input columns:

```text
prompt
completion
```

Non-training metadata:

```text
example_id
split
scenario_family
generator_version
seed
```

The completion is strict JSON. It does not contain free-form chain-of-thought.

The initial AutoScientist run will be text-only. The current public API documentation and the newer
multimodal product announcement are not fully consistent about fine-tuning image-context datasets.
MuJoCo frames and GIFs remain demo artifacts until multimodal training support is confirmed in the
entrant's workspace.

## Evaluation Gates

Dataset gate:

- deterministic regeneration from seed,
- unique example IDs,
- no family overlap between splits,
- all completions parse against the strict schema,
- all evidence steps exist in the supplied trace,
- every non-aligned patch passes executable exploit-reduction and preservation checks,
- no hidden validator result or injected defect name appears in the prompt.

Model gate:

- AutoScientist reports positive relative improvement over its base model,
- verdict macro-F1 exceeds the always-aligned baseline,
- failure-type macro-F1 improves over the base model,
- responsible-agent exact match improves over the base model,
- evidence-step F1 improves over the base model,
- JSON validity is at least 95%,
- executable patch success improves over the base model,
- results are reported on the held-out `crossing_navigation` family.

The deterministic pre-training baselines on the 640-example held-out split are:

| Baseline | Verdict accuracy | Verdict macro-F1 | Failure macro-F1 | Composite |
| --- | ---: | ---: | ---: | ---: |
| always aligned | 0.500 | 0.333 | 0.074 | 0.381 |
| reward program only | 0.500 | 0.333 | 0.593 | 0.485 |
| executable oracle ceiling | 1.000 | 1.000 | 1.000 | 1.000 |

The paired design makes the reward-program-only baseline incapable of identifying which episode
actually demonstrates the exploit.

Release gate:

- exact adapted dataset used for training is public on both platforms,
- exact adapter checkpoint is public on both platforms,
- license is compatible with the selected base model and all generated assets,
- cards disclose synthetic-data limitations and that generated patches require verification,
- public repositories load without private credentials,
- Space and Kaggle notebook run from public artifacts,
- submission links resolve in a logged-out browser.

## Implementation And Commit Plan

1. `docs: define FalsifyRL hackathon submission contract`
2. `feat: add deterministic reward-falsification core`
3. `feat: generate verified AutoScientist seed dataset`
4. `test: add leakage and executable-patch evaluation gates`
5. `feat: integrate AutoScientist adaptation and training`
6. `feat: add FalsifyRL comparison demo`
7. `docs: add Hugging Face and Kaggle release artifacts`
8. `chore: finalize challenge submission package`

Each commit must pass the smallest relevant tests. Generated model weights, secrets, raw platform
downloads, and large demo videos must not be committed to Git.

## Credential Handling

The AutoScientist API key will be requested only after the local seed dataset passes the dataset
gate. It must be supplied through an environment variable or ignored `.env` file and must never be
printed, logged, committed, or copied into a public artifact.
