from __future__ import annotations

import json
import os
from pathlib import Path

import gradio as gr

EXAMPLES = json.loads(Path("examples.json").read_text(encoding="utf-8"))
BY_ID = {example["example_id"]: example for example in EXAMPLES}
MODEL_REPO_ID = os.environ.get("MODEL_REPO_ID")
BASE_MODEL_ID = os.environ.get("BASE_MODEL_ID")
_runner = None

CSS = """
:root { --ink: #171923; --paper: #f7f5ef; --accent: #e84a5f; --safe: #168f65; }
.gradio-container { max-width: 1280px !important; background: var(--paper); }
.hero { border-left: 8px solid var(--accent); padding: 4px 0 4px 20px; }
.hero, .hero * { color: var(--ink) !important; }
#trace-badge, #trace-badge * { color: var(--ink) !important; font-weight: 700; }
.status-safe { color: var(--safe); font-weight: 700; }
.status-hack { color: var(--accent); font-weight: 700; }
"""


def parse_sections(prompt: str) -> dict[str, str]:
    headings = {
        "SCENARIO FAMILY:": "scenario",
        "TASK SPECIFICATION:": "task",
        "PROXY REWARD PROGRAM:": "reward",
        "OBSERVED EPISODE TRACE:": "trace",
    }
    result = {"instruction": []}
    active = "instruction"
    for line in prompt.splitlines():
        if line in headings:
            active = headings[line]
            result[active] = []
        else:
            result.setdefault(active, []).append(line)
    return {key: "\n".join(value).strip() for key, value in result.items()}


def trace_rows(trace: str):
    lines = [line for line in trace.splitlines() if line.strip()]
    if not lines:
        return [], []
    headers = [cell.strip() for cell in lines[0].split("|")]
    rows = [[cell.strip() for cell in line.split("|")] for line in lines[1:]]
    return headers, rows


def load_example(example_id: str):
    example = BY_ID[example_id]
    sections = parse_sections(example["prompt"])
    headers, rows = trace_rows(sections["trace"])
    role = example["case_role"]
    badge = (
        "✅ Aligned control trace"
        if role == "control"
        else "⚠️ Counterexample trace"
    )
    return (
        badge,
        sections["scenario"],
        sections["task"],
        sections["reward"],
        gr.Dataframe(headers=headers, value=rows),
        {},
        json.loads(example["gold_completion"]),
    )


def get_runner():
    global _runner
    if _runner is not None:
        return _runner
    if not MODEL_REPO_ID or not BASE_MODEL_ID:
        return None
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID)
    base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_ID,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
    )
    model = PeftModel.from_pretrained(base, MODEL_REPO_ID)
    model.eval()
    _runner = (tokenizer, model)
    return _runner


def extract_json(text: str) -> dict:
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end < start:
        return {"error": "Model output did not contain a JSON object.", "raw": text}
    try:
        return json.loads(text[start : end + 1])
    except json.JSONDecodeError as error:
        return {"error": str(error), "raw": text}


def run_critic(example_id: str):
    runner = get_runner()
    if runner is None:
        return {
            "status": "checkpoint_pending",
            "message": (
                "No model is configured. The gold verifier remains visible for preview, "
                "but is not presented as a model prediction."
            ),
        }
    tokenizer, model = runner
    prompt = BY_ID[example_id]["prompt"]
    if tokenizer.chat_template:
        model_input = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
    else:
        model_input = prompt
    inputs = tokenizer(model_input, return_tensors="pt").to(model.device)
    output = model.generate(
        **inputs,
        max_new_tokens=512,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
    )
    generated = tokenizer.decode(
        output[0, inputs["input_ids"].shape[1] :],
        skip_special_tokens=True,
    )
    return extract_json(generated)


with gr.Blocks(
    css=CSS,
    title="FalsifyRL",
    theme=gr.themes.Base(primary_hue="red", neutral_hue="slate"),
) as demo:
    gr.Markdown(
        """
        <div class="hero">
        <h1>FalsifyRL</h1>
        <p><strong>Falsify reward functions before your robots exploit them.</strong></p>
        <p>Matched reward programs. Different trajectories. Executable repairs.</p>
        </div>
        """
    )
    labels = {
        f"{example['pair_failure_type']} · {example['case_role']} · {example['example_id']}":
        example["example_id"]
        for example in EXAMPLES
    }
    selector = gr.Dropdown(
        choices=list(labels),
        value=next(iter(labels)),
        label="Held-out trace",
    )
    selected_id = gr.State(labels[next(iter(labels))])
    badge = gr.Markdown(elem_id="trace-badge")
    with gr.Row():
        scenario = gr.Textbox(label="Scenario family", interactive=False)
        task = gr.Textbox(label="True task", lines=4, interactive=False)
    reward = gr.Code(label="Proxy reward", language="python")
    trace = gr.Dataframe(label="Episode evidence", interactive=False)
    run_button = gr.Button("Run FalsifyRL critic", variant="primary")
    with gr.Row():
        prediction = gr.JSON(label="Model diagnosis")
        gold = gr.JSON(label="Executable gold diagnosis")

    def choose(label):
        example_id = labels[label]
        return (example_id, *load_example(example_id))

    selector.change(
        choose,
        inputs=selector,
        outputs=[selected_id, badge, scenario, task, reward, trace, prediction, gold],
    )
    run_button.click(run_critic, inputs=selected_id, outputs=prediction)
    demo.load(
        lambda: load_example(labels[next(iter(labels))]),
        outputs=[badge, scenario, task, reward, trace, prediction, gold],
    )


if __name__ == "__main__":
    demo.launch()
