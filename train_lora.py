import torch
from datasets import load_dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer

model_dir = "models/qwen2.5-0.5b-instruct"
output_dir = "outputs/note-lora"

train_data = load_dataset(
    "json", data_files="data/train.jsonl", split="train"
)

tokenizer = AutoTokenizer.from_pretrained(model_dir)
model = AutoModelForCausalLM.from_pretrained(
    model_dir, dtype=torch.float32
)
model.config.use_cache = False

lora = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.0,
    bias="none",
    task_type="CAUSAL_LM",
)

settings = SFTConfig(
    output_dir=output_dir,
    max_length=128,
    per_device_train_batch_size=1,
    max_steps=60,
    learning_rate=2e-4,
    logging_steps=10,
    completion_only_loss=True,
    gradient_checkpointing=False,
    bf16=False,
    fp16=False,
    optim="adamw_torch",
    dataloader_pin_memory=False,
    save_strategy="no",
    report_to="none",
)

trainer = SFTTrainer(
    model=model,
    args=settings,
    train_dataset=train_data,
    processing_class=tokenizer,
    peft_config=lora,
)

print("Training device:", trainer.args.device)
trainer.model.print_trainable_parameters()
trainer.train()
trainer.save_model(output_dir)
print("Saved LoRA adapter to:", output_dir)
