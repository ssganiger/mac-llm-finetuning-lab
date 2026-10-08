"""Generate answers from the unchanged model on held-out notes."""

import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_DIR = "models/qwen2.5-0.5b-instruct"
TEST_FILE = Path("data/test.jsonl")


def generate_answer(model, tokenizer, messages):
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

    new_ids = output_ids[0, inputs["input_ids"].shape[-1] :]
    return tokenizer.decode(new_ids, skip_special_tokens=True).strip()


def main():
    test_rows = [json.loads(line) for line in TEST_FILE.read_text(encoding="utf-8").splitlines()]
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForCausalLM.from_pretrained(MODEL_DIR, dtype=torch.float32).to("mps")
    model.eval()

    for row in test_rows:
        question = row["prompt"][0]["content"]
        note = question.split("Note: ", 1)[1]
        expected = row["completion"][0]["content"]
        answer = generate_answer(model, tokenizer, row["prompt"])
        print(f"\nNote: {note}")
        print(f"Expected: {expected}")
        print(f"Base model: {answer}")


if __name__ == "__main__":
    main()
