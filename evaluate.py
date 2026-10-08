import json
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

model_dir = "models/qwen2.5-0.5b-instruct"
adapter_dir = "outputs/note-lora"

test_rows = [
    json.loads(line)
    for line in Path("data/test.jsonl").read_text(
        encoding="utf-8"
    ).splitlines()
]

tokenizer = AutoTokenizer.from_pretrained(model_dir)
base_model = AutoModelForCausalLM.from_pretrained(
    model_dir, dtype=torch.float32
).to("mps")
base_model.eval()

def answer(model, messages):
    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    ).to("mps")

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=32,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    new_ids = output_ids[0, inputs["input_ids"].shape[-1]:]
    return tokenizer.decode(new_ids, skip_special_tokens=True).strip()

base_answers = [answer(base_model, row["prompt"]) for row in test_rows]

adapted_model = PeftModel.from_pretrained(base_model, adapter_dir)
adapted_model.to("mps")
adapted_model.eval()
adapted_answers = [
    answer(adapted_model, row["prompt"]) for row in test_rows
]

base_matches = 0
adapted_matches = 0

for row, base, adapted in zip(
    test_rows, base_answers, adapted_answers
):
    question = row["prompt"][0]["content"]
    note = question.split("Note: ", 1)[1]
    expected = row["completion"][0]["content"]
    base_matches += base == expected
    adapted_matches += adapted == expected

    print(f"\nNote: {note}")
    print(f"Expected: {expected}")
    print(f"Base: {base}")
    print(f"With LoRA: {adapted}")

print(f"\nExact matches — base: {base_matches}/{len(test_rows)}")
print(f"Exact matches — with LoRA: {adapted_matches}/{len(test_rows)}")
