"""Show how one training example becomes model tokens; no model weights are loaded."""

import json
from pathlib import Path

from transformers import AutoTokenizer

MODEL_DIR = "models/qwen2.5-0.5b-instruct"
TRAIN_FILE = Path("data/train.jsonl")


def main():
    first_line = TRAIN_FILE.read_text(encoding="utf-8").splitlines()[0]
    example = json.loads(first_line)
    messages = example["prompt"] + example["completion"]
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    formatted = tokenizer.apply_chat_template(messages, tokenize=False)
    token_ids = tokenizer.encode(formatted, add_special_tokens=False)

    print("Formatted chat:\n")
    print(formatted)
    print("Token count:", len(token_ids))
    print("First 20 token IDs:", token_ids[:20])


if __name__ == "__main__":
    main()
