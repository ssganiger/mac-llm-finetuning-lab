# Concepts behind this exercise

Start with one of the repository's training records:

```text
User: Classify this note as WORK, PERSONAL, or SHOPPING.
      Reply only in the format CATEGORY: <category>.
      Note: Email the weekly report to my manager.
Desired assistant reply: CATEGORY: WORK
```

The goal is to make the model more likely to produce the desired reply for similar notes. The 18 training records provide examples; three different records are reserved for checking the result.

## The pieces

| Term | Meaning in this project |
| --- | --- |
| Base model | Qwen2.5-0.5B-Instruct before this exercise's training. It already knows language patterns and can generate text. |
| Parameter or weight | A number inside the model that affects its predictions. The base model has hundreds of millions of them. |
| Token | A unit of text the model predicts. It may be a word, part of a word, punctuation, or a special chat marker. |
| Tokenizer | Converts text to token IDs and token IDs back to text. `inspect_tokens.py` shows both representations. |
| Chat template | Adds markers for system, user, and assistant messages, matching the format Qwen expects. |
| Prompt | The question shown to the model: the category instructions and a note. |
| Completion | The target assistant answer, such as `CATEGORY: WORK`. |
| Dataset | A collection of prompt-completion records, stored here as JSONL (one JSON object per line). |
| Forward pass | The model processes prompt tokens and estimates probabilities for subsequent answer tokens. |
| Loss | A number measuring how poorly the model predicted the desired answer tokens. Lower training loss means better fit to the shown examples. |
| Gradient / backpropagation | A calculation of how each trainable weight affects the loss. |
| Optimizer | Uses gradients to update the trainable weights. The learning rate controls update size. |
| Training step | One optimizer update. This tutorial runs 60 steps. |
| Epoch | One pass through the dataset. With 18 examples and batch size 1, 60 steps are roughly 3.33 epochs. |
| Inference | Generating an answer without updating weights. `baseline.py` and `evaluate.py` perform inference. |
| Held-out test | Examples kept out of training and used afterward to check behavior. |
| Overfitting | Fitting the training examples without improving enough on new examples. A small test set cannot rule this out. |

## What LoRA changes

In full fine-tuning, many or all base-model parameters change. In **LoRA**, the base parameters remain fixed. Small trainable matrices are added to selected model layers; their contribution adjusts the layer's output. `train_lora.py` attaches them to the `q_proj` and `v_proj` attention projections with rank `r=8`.

The recorded run reported **540,672 trainable parameters out of 494,573,440 total (0.1093%)**. The output is an **adapter**, saved in `outputs/note-lora`. To use it, `evaluate.py` loads the original base model and attaches the adapter with PEFT. The adapter is not a complete standalone model.

**PEFT** is Hugging Face's parameter-efficient fine-tuning library, which provides LoRA. **TRL** provides `SFTTrainer`, which runs this supervised fine-tuning process. TRL can run other training methods too; this exercise uses only SFT. **Transformers** loads Qwen and its tokenizer, while **PyTorch** performs tensor calculations and gradients on the Mac's **MPS** GPU backend.

## What is actually learned

This is a **conversational prompt-completion** dataset. TRL applies the model's chat template and, with `completion_only_loss=True`, measures training loss on the desired assistant response rather than on the wording of the user prompt. The model sees examples of both a category decision and the exact `CATEGORY: LABEL` answer format.

After training, the adapter returned the target format for all three held-out notes in the recorded run. The original model was not rewritten. The saved adapter changes its predictions when loaded together with that base model.

The 18/3 split is designed for learning, not for a dependable classifier. A real application would need more representative data, consistent labeling rules for ambiguous notes, a larger held-out set, and comparison against a prompting-only baseline.

## How the later MLX exercise relates

The planned `gpt-oss-20b` exercise will still use examples, tokenization, a forward pass, loss, gradients, and a LoRA adapter. It will use **MLX-LM** instead of the PyTorch/Transformers/TRL training path. Both paths run on the Mac's physically unified memory; MLX's model implementation and quantized-weight handling are tailored to Apple Silicon. That larger-model procedure will be documented only after it has been executed and checked.

## Primary documentation

- [TRL dataset formats](https://huggingface.co/docs/trl/dataset_formats)
- [TRL SFT Trainer](https://huggingface.co/docs/trl/sft_trainer)
- [TRL with PEFT](https://huggingface.co/docs/trl/peft_integration)
- [Transformers chat templates](https://huggingface.co/docs/transformers/main/chat_templating)
- [Transformers on Apple Silicon](https://huggingface.co/docs/transformers/perf_train_special)
