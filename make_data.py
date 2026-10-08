import json
from pathlib import Path

train_examples = [
    ("Email the weekly report to my manager.", "WORK"),
    ("Prepare slides for Monday's meeting.", "WORK"),
    ("Fix the login bug in our app.", "WORK"),
    ("Review the team's pull request.", "WORK"),
    ("Schedule a call with the client.", "WORK"),
    ("Update the project timeline.", "WORK"),
    ("Wish my sister a happy birthday.", "PERSONAL"),
    ("Book a dentist appointment.", "PERSONAL"),
    ("Go for a run in the evening.", "PERSONAL"),
    ("Call my friend Ravi tonight.", "PERSONAL"),
    ("Plan a weekend hike.", "PERSONAL"),
    ("Read a novel before bed.", "PERSONAL"),
    ("Purchase tomatoes and onions.", "SHOPPING"),
    ("Order a new phone charger.", "SHOPPING"),
    ("Get soap and toothpaste from the store.", "SHOPPING"),
    ("Buy a birthday gift for my sister.", "SHOPPING"),
    ("Pick up coffee beans.", "SHOPPING"),
    ("Replace the empty shampoo bottle.", "SHOPPING"),
]

test_examples = [
    ("Buy milk and bread on the way home.", "SHOPPING"),
    ("Review the project proposal before Friday.", "WORK"),
    ("Call Mom on Sunday.", "PERSONAL"),
]

def make_record(note, label):
    question = (
        "Classify this note as WORK, PERSONAL, or SHOPPING. "
        "Reply only in the format CATEGORY: <category>.\n"
        f"Note: {note}"
    )
    return {
        "prompt": [{"role": "user", "content": question}],
        "completion": [{"role": "assistant", "content": f"CATEGORY: {label}"}],
    }

def write_jsonl(path, examples):
    with path.open("w", encoding="utf-8") as file:
        for note, label in examples:
            file.write(json.dumps(make_record(note, label)) + "\n")

data_dir = Path("data")
data_dir.mkdir(exist_ok=True)
write_jsonl(data_dir / "train.jsonl", train_examples)
write_jsonl(data_dir / "test.jsonl", test_examples)

print(f"Training examples: {len(train_examples)}")
print(f"Held-out test examples: {len(test_examples)}")
print("First training record:")
print(json.dumps(make_record(*train_examples[0]), indent=2))
