# Recorded learning run

This record comes from one learner-run exercise on **2026-10-08**. The scripts and data are included in the repository; the downloaded model and generated adapter are local outputs and are not committed.

## Environment and data

| Item | Recorded value |
| --- | --- |
| Mac | MacBook Pro, Apple M3 Pro, 36 GB unified memory |
| macOS / Python | macOS 26.6.2 / Python 3.11.7 |
| Compute backend | PyTorch MPS (`MPS available: True`; training device `mps`) |
| Base model | `Qwen/Qwen2.5-0.5B-Instruct` from Hugging Face |
| Training data | 18 short notes, six each for WORK, PERSONAL, SHOPPING |
| Held-out test | Three different notes, one per category |
| Training method | TRL supervised fine-tuning with PEFT LoRA, fp32 base model |
| Trainable weights | 540,672 / 494,573,440 (0.1093%) |
| Training | 60 update steps, 3.333 epochs, 11.09 seconds reported runtime |
| Final reported training loss | 0.0964 |

Training loss reports at each ten-step interval were approximately **0.2121, 0.09892, 0.16, 0.04243, 0.02367, 0.0413**. They fluctuated, which is normal for this small set. The first fully formatted training record contained **68 tokens**.

## Held-out comparison

`evaluate.py` asked the **same prompt** of the base model and then the base model with the trained adapter. Generation was deterministic (`do_sample=False`).

| Held-out note | Expected | Base model | With LoRA |
| --- | --- | --- | --- |
| Buy milk and bread on the way home. | `CATEGORY: SHOPPING` | `SHOPPING: FOOD AND BEVERAGE` | `CATEGORY: SHOPPING` |
| Review the project proposal before Friday. | `CATEGORY: WORK` | `WORK: Note: Review the project proposal before Friday.` | `CATEGORY: WORK` |
| Call Mom on Sunday. | `CATEGORY: PERSONAL` | A multi-line answer mentioning WORK, PERSONAL, and Shopping | `CATEGORY: PERSONAL` |

**Exact-format matches: base 0/3; with LoRA 3/3.** The category decision and required format both matched for those three adapted outputs.

## How to interpret this

The result demonstrates that this local training run changed the model's responses on three unseen notes and that the saved adapter can be reloaded for inference. It does **not** establish general classification accuracy: three held-out examples are too few, the labels are simple, and the data do not cover ambiguous or adversarial notes. A broader evaluation would include many more notes, explicit labeling rules, repeated runs, and a prompting-only comparison.
