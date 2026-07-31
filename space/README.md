---
title: FalsifyRL
emoji: 🔬
colorFrom: indigo
colorTo: red
sdk: gradio
sdk_version: 5.49.1
app_file: app.py
pinned: false
license: mit
short_description: Falsify reward functions before your robots exploit them.
---

# FalsifyRL

Inspect reward-matched safe and exploit traces, run the AutoScientist-trained critic, and compare its
strict JSON diagnosis against the executable gold verifier.

The Space requires `BASE_MODEL_ID` and `MODEL_REPO_ID` variables after the best LoRA checkpoint is
published. Without them, it runs in transparent verifier-preview mode and never pretends that gold
labels are model predictions.

