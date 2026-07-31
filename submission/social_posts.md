# Social Post Drafts

Publish only after final metrics and public links are available. Replace every bracketed field.

## X

Robots optimize what we reward—not what we meant. 🔬

I built FalsifyRL with @adaption_ai AutoScientist: a critic that detects multi-agent RL reward hacks,
cites trace evidence, and proposes executable repairs.

Demo: [SPACE_URL]
Results: [EVALUATION_URL]

## LinkedIn

What happens when two robots discover that the easiest way to maximize reward is to ignore the task?

For the Adaption AutoScientist Challenge, I built **FalsifyRL**: an evidence-grounded critic for
reward hacking in embodied, multi-agent reinforcement learning.

Tag the official **Adaption** LinkedIn company page when publishing this post.

FalsifyRL reads a task specification, declarative reward function, and episode trace. It returns a
strict JSON diagnosis containing the failure type, responsible agents, evidence steps, a
counterexample configuration, and an executable reward patch.

The dataset has 1,920 reward-matched control/exploit pairs across four robotics scenario families.
Each pair shares the same reward program, so a model cannot win by recognizing suspicious code—it
must reason about what the agents actually did. Every label and patch is checked by an independent
executable task oracle.

On the held-out `crossing_navigation` family:

- base-model composite: [BASE_COMPOSITE]
- AutoScientist model composite: [TRAINED_COMPOSITE]
- JSON validity: [JSON_VALIDITY]
- executable patch success: [PATCH_SUCCESS]

Interactive demo: [SPACE_URL]
Dataset: [HF_DATASET_URL]
Model: [HF_MODEL_URL]
Code and evaluation: [GITHUB_URL]

#AutoScientist #ReinforcementLearning #MultiAgentSystems #Robotics #PhysicalAI

## Demo video storyboard

1. Show the same free-riding reward program on both sides.
2. Play the aligned control trace, then the exploit trace where one agent remains idle.
3. Reveal the critic's responsible-agent and evidence-step output.
4. Apply the predicted patch and show the exploit return drop.
5. End on the held-out base-versus-trained metrics and public artifact links.
