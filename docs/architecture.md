# Mermaid maps: Qwen + Transformers + TRL

These diagrams follow the code and 18/3 note split in this repository. Green nodes prepare data, blue nodes hold the model or framework, orange nodes update weights, purple nodes show the saved result, and red marks the Mac hardware.

## 1. From notes to a tested LoRA adapter

```mermaid
graph TD
    A["data/train.jsonl<br/>18 notes with desired CATEGORY replies"] --> B["Qwen chat template<br/>system, user, assistant markers"]
    B --> C["Tokenizer<br/>text becomes token IDs"]
    C --> D["TRL SFTTrainer and data collator<br/>batch 1, completion-only labels"]
    D --> E["Forward pass<br/>predict answer tokens"]
    F["Qwen2.5-0.5B base<br/>frozen float32 weights"] --> E
    G["PEFT LoRA<br/>trainable q_proj and v_proj"] --> E
    E --> H["Completion loss<br/>prediction versus CATEGORY target"]
    H --> I["PyTorch backpropagation<br/>gradients through MPS"]
    I --> J["Optimizer update<br/>60 steps, learning rate 2e-4"]
    J --> G
    G --> K["Save outputs/note-lora<br/>540,672 trainable parameters"]
    T["data/test.jsonl<br/>3 unseen notes"] --> V["Same prompts for base and adapter<br/>exact-format comparison"]
    F --> V
    K --> V
    V --> R["Recorded result<br/>base 0/3, adapted 3/3"]

    classDef data fill:#DCFCE7,stroke:#16A34A,color:#14532D,stroke-width:2px;
    classDef model fill:#DBEAFE,stroke:#2563EB,color:#1E3A8A,stroke-width:2px;
    classDef train fill:#FFEDD5,stroke:#EA580C,color:#7C2D12,stroke-width:2px;
    classDef result fill:#F3E8FF,stroke:#9333EA,color:#581C87,stroke-width:2px;
    class A,B,C,D,T data;
    class E,F,G model;
    class H,I,J train;
    class K,V,R result;
```

The test notes never contribute gradients. The 3/3 score describes these three notes in one run, not performance on every possible note.

## 2. What occupies the Mac during training

```mermaid
graph LR
    subgraph FILES["On disk"]
        HF["Hugging Face model download<br/>models/qwen2.5-0.5b-instruct"]
        DS["Small JSONL files<br/>18 train and 3 test notes"]
        OUT["Saved LoRA adapter<br/>outputs/note-lora"]
    end

    subgraph SOFTWARE["Python environment"]
        TF["Transformers<br/>loads model and tokenizer"]
        TR["TRL SFTTrainer<br/>runs supervised training"]
        PF["PEFT<br/>attaches LoRA to q_proj and v_proj"]
        PT["PyTorch MPS<br/>forward, backward, optimizer"]
    end

    subgraph MAC["Apple Silicon MacBook Pro used in this run"]
        UM["36 GB unified memory<br/>shared physical CPU and GPU pool"]
        BW["Frozen Qwen base weights<br/>494,573,440 total parameters"]
        LW["Trainable LoRA weights<br/>540,672 parameters, 0.1093%"]
    end

    HF --> TF --> BW --> UM
    DS --> TR --> PT --> UM
    PF --> LW --> PT
    TF --> PF
    TR --> PF
    LW --> OUT

    classDef file fill:#DCFCE7,stroke:#16A34A,color:#14532D,stroke-width:2px;
    classDef framework fill:#DBEAFE,stroke:#2563EB,color:#1E3A8A,stroke-width:2px;
    classDef adapter fill:#FFEDD5,stroke:#EA580C,color:#7C2D12,stroke-width:2px;
    classDef hardware fill:#FEE2E2,stroke:#C74634,color:#7F1D1D,stroke-width:3px;
    class HF,DS,OUT file;
    class TF,TR,PF,PT,BW framework;
    class LW adapter;
    class UM hardware;
```

`outputs/note-lora` stores the learned adjustments; it still needs the downloaded base model for inference. The 36 GB figure is the configuration used in the recorded run, not a measured minimum.
