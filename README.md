# Fine-tuning a small language model on an Apple Silicon Mac

This is a hands-on learning lab. You clone the code, run each command yourself, inspect the intermediate results, and learn what each stage does. The completed exercise uses **Hugging Face Transformers + TRL + PEFT/LoRA** to teach a small Qwen model to classify notes in a fixed format. A later exercise will apply the same ideas to **gpt-oss-20b with MLX**; that exercise is not yet included or claimed as tested.

The included run used 18 training notes and three held-out notes. Exact-format matches changed from **0/3 with the base model to 3/3 with its LoRA adapter** on one M3 Pro Mac. This is a demonstration of the workflow with a tiny dataset, not a reliable estimate of broader model quality. See [the recorded results](RESULTS.md).

## What you will learn

1. Download a public model from the Hugging Face Hub.
2. Read a prompt-completion dataset and see how a chat template and tokenizer transform it.
3. Record the base model's answers before training.
4. Train a LoRA adapter with TRL's `SFTTrainer` on Apple's MPS GPU.
5. Load the saved adapter and compare answers on examples excluded from training.

The model, tokenizer, dataset, training loop, and adapter are separate pieces. [Training concepts](docs/concepts.md) explains each one, and [workflow diagrams](docs/architecture.md) show how they connect.

## Mac configuration

| Item | Practical target for this tutorial | What was tested |
| --- | --- | --- |
| Mac | Apple Silicon, M1 or newer, with MPS available | MacBook Pro with M3 Pro |
| Unified memory | **16 GB suggested starting point**; this lower target has **not** been tested for this exact script | 36 GB |
| macOS | macOS 14 or newer for the exact pinned PyTorch wheel used here | macOS 26.6.2 |
| Python | Python 3.11 | Python 3.11.7 |
| Free disk space | **10 GB suggested allowance** for packages, model, caches, and outputs; usage varies | The model download was about 1 GB and the virtual environment about 1.3 GB |
| Internet | Needed once for Python packages and the public model | Public model downloaded without Hugging Face login |

These are practical starting requirements, **not a measured minimum**. Hugging Face states that MPS itself needs macOS 12.3+ and that the model must fit in unified memory; this repository uses a newer pinned PyTorch wheel that was downloaded as `macosx_14_0_arm64`. [Hugging Face Apple Silicon guide](https://huggingface.co/docs/transformers/perf_train_special). If `torch.backends.mps.is_available()` is false, stop before training and check the Mac, macOS, and PyTorch install.

## Tools used

| Tool | Version in the recorded run | Why it is here |
| --- | --- | --- |
| Git | Your installed version | Clone this repository and track source changes. |
| Python and `venv` | 3.11.7 | Run the examples in an isolated environment. |
| `pip` | Version may vary | Install the pinned Python packages. |
| Hugging Face Hub CLI (`hf`) | `huggingface-hub` 1.33.0 | Download the public model into `models/`. |
| Qwen2.5-0.5B-Instruct | Public 0.5B-parameter model | The small base model for the learning exercise. |
| PyTorch + Apple MPS | 2.14.1 | Run model calculations and gradients on the Mac GPU. |
| Transformers | 5.19.0 | Load the model and tokenizer and generate answers. |
| Datasets | 5.1.0 | Read the JSONL training examples. |
| TRL | 1.14.2 | Run supervised fine-tuning with `SFTTrainer`. |
| PEFT | 0.21.2 | Add, save, and reload LoRA adapter weights. |
| Accelerate | 1.15.0 | Support device placement in the training stack. |
| MLX-LM | Planned for the **later** gpt-oss-20b exercise | Not needed for this completed small-model exercise. |

All pinned Python versions appear in [`requirements.txt`](requirements.txt). The [Qwen model card](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct) describes the model and its Apache 2.0 license. This repository's tutorial code is MIT licensed; the model weights are downloaded separately under their own license.

## Files you get when you clone

| File | Purpose |
| --- | --- |
| [`data/train.jsonl`](data/train.jsonl) | 18 examples shown to the model during training, six per category. |
| [`data/test.jsonl`](data/test.jsonl) | Three different examples reserved for after-training checks. |
| [`make_data.py`](make_data.py) | Shows exactly how those two data files were built; optional to run. |
| [`inspect_tokens.py`](inspect_tokens.py) | Displays one example with chat markers and token IDs. |
| [`baseline.py`](baseline.py) | Generates answers from the unchanged model on the held-out notes. |
| [`train_lora.py`](train_lora.py) | Trains and saves the LoRA adapter. |
| [`evaluate.py`](evaluate.py) | Compares base and adapted answers on identical held-out prompts. |

The large model download, `.venv`, generated adapter, and personal session log are deliberately excluded from Git. Readers create their own local copies during the steps below.

## Follow the exercise yourself

Run these steps in order in **Terminal on your Mac**. Each command is separate so you can inspect what happened before moving on. Commands shown with a GitHub URL assume this repository is published at the URL below.

### 1. Get the tutorial code

```bash
git clone https://github.com/ssganiger/mac-llm-finetuning-lab.git
cd mac-llm-finetuning-lab
```

**Why:** `git clone` copies the small source files and example data from GitHub; `cd` makes this folder the working directory. It does not download the 1 GB model or run training. Inspect the files before executing them.

### 2. Make an isolated Python environment

```bash
python3 --version
python3 -m venv .venv
source .venv/bin/activate
python --version
```

**Why:** The version checks show which Python you are using. `python3 -m venv .venv` creates a project-local Python environment, and `source` makes this Terminal session use it. The prompt should show `(.venv)` and Python 3.11.x. The environment keeps this project's packages separate from other projects.

### 3. Install the libraries

```bash
python -m pip install -r requirements.txt
```

**Why:** `python -m pip` uses the installer inside `.venv`. `-r requirements.txt` installs the versions recorded in the successful run. This step downloads software packages, **not the model**. If a pinned version is unavailable for your Python/macOS combination, record the error before changing versions; results may differ with another stack.

### 4. Check Apple GPU access

```bash
python -c 'import torch; print("MPS available:", torch.backends.mps.is_available())'
```

**Why:** MPS is PyTorch's Apple GPU backend. This read-only check should print `MPS available: True`. The training script also prints its selected device before updating weights.

### 5. Download the base model

```bash
hf download Qwen/Qwen2.5-0.5B-Instruct --local-dir models/qwen2.5-0.5b-instruct
```

**Why:** `hf download` retrieves the public model weights, tokenizer, and configuration from the Hugging Face Hub. `--local-dir` puts them where the scripts expect them. The download is about 1 GB. This is still the **unchanged base model**; no training happens. The command may show a harmless warning about unauthenticated requests and rate limits. [Hugging Face CLI guide](https://huggingface.co/docs/huggingface_hub/guides/cli).

### 6. Understand one data record

Open [`data/train.jsonl`](data/train.jsonl) and [`data/test.jsonl`](data/test.jsonl) in any text editor. Each line is one JSON object with a `prompt` (user message) and a `completion` (desired assistant message). One example looks like:

```json
{"prompt": [{"role": "user", "content": "Classify this note as WORK, PERSONAL, or SHOPPING. Reply only in the format CATEGORY: <category>.\nNote: Email the weekly report to my manager."}], "completion": [{"role": "assistant", "content": "CATEGORY: WORK"}]}
```

**Why:** Supervised fine-tuning learns from desired responses. The three `test.jsonl` notes are withheld from training so we can check behavior on unseen examples. The dataset is intentionally tiny and balanced across the three categories. [`make_data.py`](make_data.py) shows how it was made; running `python make_data.py` is optional and rewrites both data files with the same examples. [TRL dataset format guide](https://huggingface.co/docs/trl/dataset_formats).

### 7. Watch tokenization

```bash
python inspect_tokens.py
```

**Why:** This reads the first training record, applies Qwen's chat template, and prints its token count and first 20 token IDs. It does not load model weights or train. In the recorded run, the first complete conversation was **68 tokens** and included system, user, and assistant markers. `train_lora.py` allows up to 128 tokens per example.

### 8. Record the base model's answers

```bash
python baseline.py
```

**Why:** This loads the downloaded model onto MPS and asks the three held-out questions **before** any adapter is applied. It prints expected answers only as a reference; the model does not see those expected answers. `torch.no_grad()` prevents training calculations. This gives you a fair baseline using the same prompts and `float32` model precision as the later comparison.

### 9. Train a LoRA adapter

Read [`train_lora.py`](train_lora.py), then run:

```bash
python train_lora.py
```

**Why:** `SFTTrainer` formats and tokenizes the 18 training examples, runs a forward pass, measures loss on the desired completion tokens, backpropagates gradients, and updates only LoRA weights. The base model stays frozen. The script makes **60 update steps**, prints loss every 10 steps, and saves the adapter to `outputs/note-lora`. It does not upload anything to the Hub.

| Setting in the script | Value | Reason |
| --- | --- | --- |
| `r` / `lora_alpha` | 8 / 16 | Small adapter capacity for a toy task. |
| `target_modules` | `q_proj`, `v_proj` | Add trainable LoRA weights to two attention projections. |
| `max_length` | 128 tokens | Fits the short examples; the inspected example was 68 tokens. |
| Batch size | 1 | Low memory demand and simple step counting. |
| `max_steps` | 60 | A little more than three passes over 18 examples. |
| Learning rate | `2e-4` | Size of LoRA weight updates. |
| `completion_only_loss` | `True` | Learn the desired answer rather than the wording of the question. |
| Model `dtype` | `float32` | Simple, stable training for this small model; this run is **LoRA**, not QLoRA. |
| `save_strategy` | `no` | Save the final adapter at the end, without intermediate checkpoints. |

Look for `Training device: mps`, a small trainable-parameter percentage, 60/60 steps, and `Saved LoRA adapter to: outputs/note-lora`. In the recorded run, **540,672 of 494,573,440 parameters (0.1093%)** were trainable. Loss can fluctuate between reports; it measures fit to training examples, not held-out quality. [TRL SFT guide](https://huggingface.co/docs/trl/sft_trainer); [TRL PEFT guide](https://huggingface.co/docs/trl/peft_integration).

### 10. Compare before and after on held-out notes

```bash
python evaluate.py
```

**Why:** This loads the base model, records its answers on `data/test.jsonl`, attaches the saved adapter, and asks the **identical prompts** again. Generation uses `do_sample=False` so results are repeatable. It prints expected, base, and adapted answers plus exact-format match counts. This is inference only; no weights are updated. In the recorded run, the count changed from **0/3 to 3/3**. [TRL use-model guide](https://huggingface.co/docs/trl/use_model).

### 11. Inspect what was saved

```bash
ls -lh outputs/note-lora/adapter_model.safetensors
```

**Why:** `ls -lh` shows the size of the adapter file in human-readable units. The adapter is much smaller than the downloaded base model because it contains only the learned LoRA weights. It needs the base model to generate answers.

## Reading the result carefully

The test set contains only three examples. The exact-match result shows that this adapter followed the specified format on those three held-out notes in one run. A stronger claim would need a larger, more varied test set, repeated runs, and error analysis. Some tasks can also be handled with clear prompting alone; this lab focuses on seeing the mechanics of training and measuring the change.

If you change the labels or dataset, keep training and test examples separate. Re-run `baseline.py`, `train_lora.py`, and `evaluate.py` in that order, and treat any new result as a separate experiment.

## If something stops

| Symptom | What to check |
| --- | --- |
| `MPS available: False` | Confirm Apple Silicon, supported macOS, active `.venv`, and MPS-enabled PyTorch. |
| `hf: command not found` | Confirm `.venv` is active and `python -m pip install -r requirements.txt` completed. |
| Missing `models/qwen2.5-0.5b-instruct` | Complete step 5 from the project root. |
| Missing `outputs/note-lora` | Training has not completed and saved the adapter; inspect the training error first. |
| Out-of-memory error | Close other memory-heavy apps and confirm the model is the 0.5B version. A smaller batch is already used. |
| Loss is `nan` or `inf` | Stop and keep the full log; check package versions and training settings before interpreting results. |

## Next exercise: gpt-oss-20b with MLX

The planned second exercise will fine-tune `gpt-oss-20b` locally with MLX-LM and a Hugging Face model copy. It will reuse the concepts here: training examples, chat formatting, a frozen base, LoRA weights, and held-out comparison. An Ollama `gpt-oss:20b` download is a separate inference-format copy. The MLX procedure will be added after it is run and verified on the Mac; this repository currently contains **only the completed Qwen/TRL exercise**.
