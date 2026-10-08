# Visual map of the learning path

## Exercise 1: the completed Qwen + TRL training loop

```mermaid
graph TD
    A["18 training notes<br/>prompt + desired CATEGORY reply"] --> B["Qwen chat template<br/>system, user, assistant markers"]
    B --> C["Tokenizer<br/>text → token IDs"]
    C --> D["TRL data collator<br/>batch and answer labels"]
    D --> E["Forward pass<br/>base model + LoRA adapter"]
    F["Frozen Qwen base weights"] --> E
    G["Trainable LoRA weights<br/>q_proj and v_proj"] --> E
    E --> H["Loss on desired reply tokens"]
    H --> I["PyTorch backpropagation<br/>gradients on MPS"]
    I --> J["Optimizer update<br/>60 steps"]
    J --> G
    G --> K["Saved adapter<br/>outputs/note-lora"]
    K --> L["Three held-out notes<br/>compare base vs adapted"]
    M["Three test notes<br/>never used for updates"] --> L

    classDef data fill:#DCFCE7,stroke:#16A34A,color:#14532D,stroke-width:2px;
    classDef model fill:#DBEAFE,stroke:#2563EB,color:#1E3A8A,stroke-width:2px;
    classDef train fill:#FFEDD5,stroke:#EA580C,color:#7C2D12,stroke-width:2px;
    classDef result fill:#F3E8FF,stroke:#9333EA,color:#581C87,stroke-width:2px;
    class A,B,C,D,M data;
    class E,F,G model;
    class H,I,J train;
    class K,L result;
```

## Two frameworks on one Apple Silicon Mac

The left side is the **verified exercise in this repository**. The right side is the **planned next exercise**; its code and results will be added after the learner runs it.

```mermaid
graph LR
    subgraph SMALL["Completed: small Qwen model"]
        direction TB
        SH["Hugging Face Hub<br/>Qwen2.5-0.5B-Instruct"] --> ST["Transformers<br/>model + tokenizer"]
        ST --> TR["TRL SFTTrainer<br/>PyTorch / MPS"]
        SP["PEFT LoRA adapter<br/>base weights frozen"] --> TR
        TR --> SA["Saved small-model adapter"]
        SA --> SE["Held-out evaluation<br/>0/3 base → 3/3 adapted"]
    end

    subgraph LARGE["Planned: gpt-oss-20b"]
        direction TB
        GH["Hugging Face Hub<br/>gpt-oss-20b model copy"] --> GM["MLX-LM<br/>Apple Silicon training"]
        GQ["Compact model weights<br/>MXFP4 / MLX quantization"] --> GM
        GL["LoRA adapter<br/>base weights frozen"] --> GM
        GM --> GE["Held-out comparison<br/>result pending"]
    end

    subgraph MAC["Mac hardware"]
        UM["Apple Silicon unified memory<br/>shared physical pool for CPU and GPU"]
    end

    TR --> UM
    GM --> UM

    classDef hub fill:#F3E8FF,stroke:#9333EA,color:#581C87,stroke-width:2px;
    classDef framework fill:#DBEAFE,stroke:#2563EB,color:#1E3A8A,stroke-width:2px;
    classDef adapter fill:#FFEDD5,stroke:#EA580C,color:#7C2D12,stroke-width:2px;
    classDef result fill:#DCFCE7,stroke:#16A34A,color:#14532D,stroke-width:2px;
    classDef hardware fill:#FEE2E2,stroke:#C74634,color:#7F1D1D,stroke-width:3px;
    class SH,GH hub;
    class ST,TR,GM,GQ framework;
    class SP,SA,GL adapter;
    class SE,GE result;
    class UM hardware;
```

An Ollama `gpt-oss:20b` installation is a separate inference copy. The later MLX training path will obtain compatible weights from Hugging Face.
