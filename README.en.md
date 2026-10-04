# Qwen2.5 LoRA Instruction Fine-Tuning

A complete pipeline for instruction fine-tuning of Qwen2.5 using LoRA, covering data preparation, training, evaluation, and quantization. Designed to run on free Google Colab T4 GPU.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/lhh737/Qwen-Lora-finetune/blob/main/finetune_qwen.ipynb)

[中文](./README.md) | **English**

---

## Table of Contents

- [Overview](#overview)
- [Pipeline](#pipeline)
- [Environment Setup](#environment-setup)
- [Model Selection](#model-selection)
- [Fine-Tuning Method](#fine-tuning-method)
- [Data Preparation](#data-preparation)
- [LoRA Configuration](#lora-configuration)
- [Training](#training)
- [Evaluation](#evaluation)
- [Quantization](#quantization)
- [Project Structure](#project-structure)
- [Results](#results)
- [Production Gap Analysis](#production-gap-analysis)
- [References](#references)

---

## Overview

**Problem**: Open-source large language models often underperform in domain-specific tasks. General-purpose models lack knowledge of specialized terminology and fail to follow domain-specific response patterns.

**Solution**: Parameter-efficient fine-tuning using LoRA (Low-Rank Adaptation). This approach adapts a pre-trained model to a target domain with minimal computational cost — approximately 0.1% of the parameters are trainable, making it feasible on consumer GPUs.

**Scope**: This project demonstrates the complete lifecycle:
1. Data acquisition and formatting
2. Model and method selection rationale
3. LoRA-based fine-tuning
4. Quantitative and qualitative evaluation
5. Model quantization for deployment

**Constraints**: Zero monetary cost (Google Colab free tier), single T4 GPU (16GB VRAM), 3000 training samples.

---

## Pipeline

```
Data Preparation          Training                  Evaluation               Quantization
─────────────────         ──────────                ───────────              ─────────────
alpaca-zh dataset         Qwen2.5 base model        Base vs LoRA output      Merge LoRA weights
  ↓                       ↓                           ↓                        ↓
ChatML formatting         LoRA adapter injection     20 QA test set            GPTQ/AWQ INT4
  ↓                       ↓                           ↓                        ↓
3000 samples              SFTTrainer training        Keyword hit-rate         Quantized model
                          (2 epochs, ~8 min)          evaluation               (~60% size reduction)
```

---

## Environment Setup

### Platform Comparison

| Platform | GPU | VRAM | Cost | Session Limit | Suitability |
|----------|-----|------|------|---------------|-------------|
| **Google Colab (Free)** | T4 | 16 GB | ¥0 | ~1 hour | This project |
| Google Colab (Pro) | T4/V100/A100 | 16-40 GB | ~¥80/mo | ~24 hours | Larger models |
| Kaggle Notebooks | T4/P100 | 16 GB | ¥0 | 30h/week | Alternative |
| AutoDL | 3090/4090/A100 | 24-80 GB | ¥2-15/hr | None | Production |
| Local RTX GPU | 3060-4090 | 6-24 GB | Existing hardware | None | Small experiments |

**Selection Rationale**: Google Colab free tier satisfies all project requirements:
- T4 16GB VRAM is sufficient for 1.5B LoRA (~6GB) and 7B QLoRA (~12GB)
- Pre-configured PyTorch/CUDA environment eliminates setup overhead
- Zero cost aligns with the learning-oriented nature of this project
- Training time for 1.5B (~8 minutes) is well within the 1-hour session limit

**Limitations**: T4 is approximately 3-4x slower than RTX 3090. Models larger than 7B cannot fit in 16GB VRAM without aggressive quantization.

### Quick Start

```bash
# 1. Clone repository
git clone https://github.com/lhh737/Qwen-Lora-finetune.git
cd Qwen-Lora-finetune

# 2. Open in Colab
#    - Visit https://colab.research.google.com
#    - Upload finetune_qwen.ipynb
#    - Runtime → Change runtime type → T4 GPU
#    - Execute cells sequentially (Shift+Enter)
```

Expected runtime by model size:

| Model | Method | Runtime |
|-------|--------|---------|
| Qwen2.5-1.5B | LoRA (FP16) | ~8 minutes |
| Qwen2.5-7B | QLoRA (4-bit) | ~40 minutes |

---

## Model Selection

### Open-Source LLM Landscape (2025-2026)

| Model | Developer | Chinese | English | Code | Ecosystem |
|-------|-----------|---------|---------|------|-----------|
| **Qwen2.5** | Alibaba | ★★★★☆ | ★★★★☆ | ★★★★☆ | ★★★★☆ |
| LLaMA 3.1 | Meta | ★★★☆☆ | ★★★★★ | ★★★★☆ | ★★★★★ |
| DeepSeek-V3/R1 | DeepSeek | ★★★★★ | ★★★★★ | ★★★★★ | ★★★☆☆ |
| GLM-4 | Zhipu AI | ★★★★★ | ★★★☆☆ | ★★★☆☆ | ★★★☆☆ |
| Yi 1.5 | 01.AI | ★★★★☆ | ★★★★☆ | ★★★☆☆ | ★★★☆☆ |

### Selection Criteria

Three factors determined the choice of Qwen2.5:

**1. Chinese Tokenization Efficiency**

Qwen2.5 uses a vocabulary of 151,642 tokens with extensive Chinese character coverage. For Chinese text, this results in fewer tokens per character compared to LLaMA 3, which was primarily optimized for English. The tokenizer dimension directly impacts both training and inference efficiency.

**2. HuggingFace Ecosystem Compatibility**

Qwen2.5 follows standard Transformers library interfaces:

```python
# Uniform loading interface across all model sizes
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-1.5B",     # Change ID to switch model size
    torch_dtype="auto",
    device_map="auto",
    trust_remote_code=True,
)
```

This guarantees compatibility with PEFT, TRL, vLLM, and other ecosystem tools without adapter code.

**3. Full Model Spectrum**

The same architecture scales from 0.5B to 72B parameters, allowing experimentation at smaller sizes before scaling up. Switching from 1.5B to 7B requires only changing `MODEL_ID`.

### Recommended Configuration

| Scenario | Model | Method | VRAM |
|----------|-------|--------|------|
| First run, rapid iteration | Qwen2.5-1.5B | LoRA (FP16) | ~6 GB |
| Better response quality | Qwen2.5-7B | QLoRA (NF4) | ~12 GB |

The technical pipeline is identical regardless of model size. The choice between 1.5B and 7B is a trade-off between training speed and output quality.

---

## Fine-Tuning Method

### Comparison of Approaches

| Method | Trainable Params | VRAM (7B model) | Speed | Quality |
|--------|-----------------|-----------------|-------|---------|
| Full Fine-Tuning | 100% (7B) | ~168 GB | Slow | Best |
| LoRA (r=16) | ~0.1% (7M) | ~14 GB | Fast | Near full FT |
| QLoRA (NF4 + LoRA) | ~0.1% (7M) | ~6 GB | Moderate | Near LoRA |

### Why Not Full Fine-Tuning?

Memory requirements for full fine-tuning a 7B parameter model with FP16 precision:

| Component | Memory | Calculation |
|-----------|--------|-------------|
| Model weights | 14 GB | 7B × 2 bytes (FP16) |
| Gradients | 14 GB | 7B × 2 bytes (FP16) |
| Adam optimizer states | 56 GB | 7B × 8 bytes (FP32 momentum + variance) |
| Subtotal (without activations) | 84 GB | |
| Activations (per batch) | 20-80 GB | Model and batch-size dependent |
| **Total** | **~110-168 GB** | |

Full fine-tuning requires 4-8 A100 GPUs (80GB each), which is impractical for individual learning. LoRA reduces memory requirements by 10-20x by limiting trainable parameters to ~0.1% of the total.

### LoRA Principle

LoRA (Low-Rank Adaptation) constrains weight updates to a low-rank decomposition. Given a pre-trained weight matrix _W₀_ ∈ ℝ^(d×k):

_W = W₀ + ΔW = W₀ + BA_, where _B_ ∈ ℝ^(d×r), _A_ ∈ ℝ^(r×k), and _r_ ≪ min(d, k)

During training, _W₀_ is frozen; only _A_ and _B_ are updated. The forward pass becomes:

_h = Wx = W₀x + BAx_

This reduces trainable parameters from _d²_ to _2dr_. For a typical layer with d=4096 and r=16, this represents a 99.2% reduction in trainable parameters.

The effectiveness of LoRA stems from an empirical finding: pre-trained models have low **intrinsic rank** — meaning the effective dimensionality of task-specific updates is much smaller than the parameter space. Full-rank updates learn mostly redundant information that low-rank approximations capture adequately.

### QLoRA

QLoRA (Quantized LoRA) further reduces memory by loading the base model in 4-bit NF4 (NormalFloat4) format before applying LoRA adapters. This reduces base model memory from 14 GB to approximately 3.5 GB for a 7B model.

The trade-off is a 20-30% training speed decrease due to dequantization overhead during each forward pass.

---

## Data Preparation

### Instruction Fine-Tuning vs. Continued Pretraining

| Aspect | Instruction Fine-Tuning (SFT) | Continued Pretraining |
|--------|------------------------------|----------------------|
| Objective | Teach response format and behavior | Inject domain knowledge |
| Data format | (instruction, response) pairs | Raw text corpora |
| Data volume | Thousands to tens of thousands | Millions to billions of tokens |
| Effect | Improves instruction following | Improves knowledge coverage |

### Dataset Comparison

| Dataset | Size | Quality | Format | Access |
|---------|------|---------|--------|--------|
| **alpaca-zh** | 52K | ★★★☆ | Standard | HuggingFace |
| BELLE | 3.5M | ★★★★ | Standard | HuggingFace |
| Firefly | 1.1M | ★★★★ | Standard | HuggingFace |
| COIG | 50K | ★★★★ | Standard | HuggingFace |

**Selection**: alpaca-zh is chosen for low barrier to entry (direct HuggingFace loading), consistent format (instruction + input + output per sample), and sufficient quality for a learning project. 3000 samples are adequate for demonstrating the pipeline.

### ChatML Format

Different models require different conversational formats:

```
Qwen2.5 (ChatML):
<|im_start|>user
{question}
<|im_end|>
<|im_start|>assistant
{response}
<|im_end|>

LLaMA 3:
<|begin_of_text|><|start_header_id|>user<|end_header_id|>
{question}<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>
{response}<|eot_id|>
```

The ChatML format must match the model's pre-training format. Format mismatch can cause training to fail (no loss decrease) or produce unusable model outputs.

### Data Preparation Code

```python
from datasets import load_dataset

dataset = load_dataset("shibing624/alpaca-zh", split="train")
dataset = dataset.select(range(3000))

CHAT_TEMPLATE = """<|im_start|>user
{instruction}
<|im_end|>
<|im_start|>assistant
{output}
<|im_end|>"""

def format_chat(ex):
    return {"text": CHAT_TEMPLATE.format(
        instruction=ex["instruction"], output=ex["output"]
    )}

dataset = dataset.map(format_chat)
```

---

## LoRA Configuration

### Parameters

**Rank (r)**: Controls the dimensionality of low-rank matrices. Higher rank increases expressiveness at the cost of more trainable parameters.

```python
r=1   →  2×d×1 parameters (minimal capacity)
r=8   →  2×d×8 parameters (minimum effective)
r=16  →  2×d×16 parameters (recommended starting point)
r=32  →  2×d×32 parameters (high capacity)
r=64  →  2×d×64 parameters (diminishing returns)
```

Empirical evidence from QLoRA paper and community practice indicates that r=16 provides the best balance across most tasks. Increasing beyond 16 yields marginal improvements.

**lora_alpha**: Controls the scaling factor applied to LoRA updates:

`actual_update = (lora_alpha / r) × BA`

Setting `lora_alpha = 2r` (scaling factor of 2) is the most common configuration. The scaling compensates for the zero initialization of B, which would otherwise produce near-zero updates at the start of training.

**target_modules**: Specifies which parameter matrices receive LoRA adapters.

| Module | Layer | Purpose | Include |
|--------|-------|---------|---------|
| q_proj | Attention | Query projection | ✅ |
| k_proj | Attention | Key projection | ✅ |
| v_proj | Attention | Value projection | ✅ |
| o_proj | Attention | Output projection | ✅ |
| gate_proj | FFN | Gate | Optional |
| up_proj | FFN | Up-projection | Optional |
| down_proj | FFN | Down-projection | Optional |

Fine-tuning only attention layers is standard practice: it captures the most task-relevant adaptations while minimizing trainable parameters. Adding MLP layers doubles parameter count with limited additional benefit for most tasks.

**lora_dropout**: Dropout rate applied to LoRA layer outputs. The 0.05 default provides regularization when training on small datasets (thousands of samples).

### Default Configuration

```python
LoraConfig(
    r=16,
    lora_alpha=32,                      # 2r for scaling factor of 2
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",                        # Do not train bias terms
    task_type="CAUSAL_LM",              # Causal language modeling
)
```

---

## Training

### Code Structure

The training script (`train_lora.py`) consists of four phases:

1. **Tokenizer loading**: `AutoTokenizer.from_pretrained()` with appropriate pad_token configuration
2. **Model loading**: `AutoModelForCausalLM.from_pretrained()` with `torch_dtype="auto"` and `device_map="auto"`
3. **LoRA configuration**: `LoraConfig` with the parameters specified above
4. **Training execution**: `SFTTrainer` from the TRL library, which wraps HuggingFace `Trainer` with automatic data formatting for instruction tuning

### Key Training Arguments

| Argument | Value | Purpose |
|----------|-------|---------|
| `per_device_train_batch_size` | 4 | Samples per GPU per step |
| `gradient_accumulation_steps` | 4 | Accumulate gradients over N steps; effective batch = 4×4=16 |
| `num_train_epochs` | 2 | Training cycles over the dataset |
| `learning_rate` | 2e-4 | LoRA requires 2-4× higher LR than full FT |
| `fp16` | True | Mixed precision for VRAM efficiency |

**Effective batch size** = batch_size × gradient_accumulation = 16. Gradient accumulation enables larger effective batch sizes without exceeding VRAM limits.

**Learning rate**: LoRA typically uses 1e-4 to 5e-4, higher than full fine-tuning (1e-5 to 5e-5), because fewer parameters require larger update steps.

### Loss Interpretation

Expected loss curve for Qwen2.5-1.5B + 3000 samples + 2 epochs:

| Step | Loss | Interpretation |
|------|------|----------------|
| 0 | ~1.8 | Initial state, near-random prediction |
| 20 | ~1.2 | Rapid decrease, learning dialogue format |
| 50 | ~0.9 | Steady improvement |
| 100 | ~0.7 | Approaching convergence |
| End | ~0.5-0.6 | Converged |

Loss values are model- and dataset-dependent. Focus on the **trend** (monotonically decreasing) rather than absolute values. Loss below 0.3 may indicate overfitting; loss above 1.5 after extended training suggests configuration issues (data format, learning rate, or model loading errors).

### Common Issues

| Symptom | Likely Cause | Resolution |
|---------|-------------|------------|
| Loss stagnant (>2.0) | Data format mismatch | Verify ChatML formatting, print sample data |
| Loss < 0.1 | Overfitting | Reduce epochs, increase dropout, add data |
| Loss oscillating | Learning rate too high | Reduce to 1e-4 or 5e-5 |
| CUDA OOM | Batch size too large | Reduce per_device_train_batch_size |

---

## Evaluation

### Problem Statement

Evaluating generative language models is fundamentally different from classification tasks. Standard metrics like BLEU and ROUGE rely on n-gram overlap, which fails to capture semantic quality:

```
Question: "LoRA 微调的核心原理是什么？"

Response A: "LoRA 通过低秩分解降低参数量，冻结原权重并插入低秩矩阵"
Response B: "LoRA is Low-Rank Adaptation" (English, correct but different language)
Response C: "LoRA 是一种方法" (vague, lacks specifics)
```

BLEU would incorrectly rank A and B similarly (low n-gram overlap due to different expression), while failing to distinguish A from C (both have minimal overlap with any reference).

### Keyword Hit-Rate Method

A deterministic evaluation approach: for each test question, pre-define 3-4 domain-specific keywords that represent essential knowledge points.

| Question | Keywords |
|----------|----------|
| LoRA 微调的核心原理是什么？ | 低秩, 冻结, 插入, 参数量 |
| 什么是 KV Cache？ | 缓存, Key, Value, 加速 |

**Scoring**: `hit-rate = (number of keywords present in response) / (total keywords)`

**Rationale**: This method evaluates whether the model covers essential concepts, rather than penalizing lexical variation. It is transparent (each point maps to a specific keyword), reproducible (deterministic), and diagnostic (which concepts are consistently missed).

**Limitations** (acknowledged):
- Synonym coverage: models may express concepts without exact keyword matches
- Correctness not guaranteed: keyword presence does not confirm correct usage
- Depth not measured: shallow coverage earns the same score as deep explanation
- Mitigation: complement quantitative hit-rate with qualitative manual review

### Expected Results

```
Metric                  Base Model    Fine-Tuned    Change
─────────────────────   ──────────    ──────────    ──────
Average hit-rate         ~38%          ~65%          +71%
Questions improved       —             15/20         —
Questions regressed      —             3/20          —
```

Note: A 1.5B model has inherent capacity limitations. Upgrading to 7B would likely improve absolute scores, but the relative improvement from fine-tuning follows the same pattern.

---

## Quantization

### Motivation

Model storage requirements by precision:

| Model | FP16 | INT4 | Compression |
|-------|------|------|-------------|
| 1.5B | 3 GB | 0.8 GB | 73% |
| 7B | 14 GB | 3.5 GB | 75% |
| 14B | 28 GB | 7 GB | 75% |
| 72B | 144 GB | 36 GB | 75% |

Quantization reduces both storage footprint and inference latency (less data to transfer between memory and compute units).

### Method Comparison

| Aspect | AWQ | GPTQ |
|--------|-----|------|
| Principle | Activation-aware weight protection | Second-order Hessian optimization |
| Accuracy loss | ~0.5% | ~0.5-1% |
| Quantization speed | Fast | Moderate |
| Inference speed | Fast (vLLM native) | Moderate |
| Ecosystem | vLLM built-in | Mature toolchain |

Both methods are viable. This project supports both via `quantize.py` (default: GPTQ). AWQ is recommended for production deployment through vLLM.

---

## Project Structure

```
qwen-lora-finetune/
├── README.md                   # Chinese documentation
├── README.en.md                # English documentation (this file)
├── finetune_qwen.ipynb         # Colab notebook (primary entry point)
├── data_prep.py                # Dataset loading and ChatML formatting
├── train_lora.py               # LoRA training script (CLI with argparse)
├── evaluate.py                 # Keyword hit-rate evaluation
├── quantize.py                 # Model quantization (GPTQ/AWQ)
├── requirements.txt            # Python dependencies
├── .gitignore
└── results/                    # Evaluation outputs (after running)
```

---

## Results

*Results to be added after running on Colab.*

The expected output includes:
- Training loss curve
- Before/after evaluation comparison table
- Per-question keyword hit-rate breakdown

---

## Production Gap Analysis

### Comparison with Industrial Deployment

| Dimension | This Project | Production Environment |
|-----------|-------------|----------------------|
| **Data** | Public dataset, 3K samples, general domain | Proprietary data, 10K-100K+ samples, domain-specific |
| **Training** | Single GPU, LoRA, 2 epochs | Distributed training, multi-stage (SFT + RLHF/DPO) |
| **Evaluation** | 20-question keyword test | Comprehensive test suite, human eval, A/B testing |
| **Infrastructure** | Colab notebook | Containerized API (vLLM/TGI), load balancing, monitoring |
| **Model Management** | Local file storage | Model registry (DVC/MLflow), version control, CI/CD |
| **Monitoring** | None | Latency P50/P99, throughput, error rates, data drift |

### Subsequent Learning Path

| Direction | Actions | Resources Needed |
|-----------|---------|-----------------|
| Scale up | Run QLoRA on 7B model | AutoDL or Colab Pro |
| Deepen evaluation | Implement LLM-as-Judge | API access (e.g., GPT-4, DeepSeek) |
| Production deployment | vLLM + FastAPI + Docker | Local GPU or cloud instance |
| Foundational understanding | Read LoRA/QLoRA papers | Academic access |

---

## References

- Hu et al. "LoRA: Low-Rank Adaptation of Large Language Models." ICLR 2022.
- Dettmers et al. "QLoRA: Efficient Finetuning of Quantized LLMs." NeurIPS 2023.
- Qwen Team. "Qwen2.5 Technical Report." arXiv:2412.15115, 2024.
- HuggingFace PEFT Documentation. https://huggingface.co/docs/peft
- HuggingFace TRL Documentation. https://huggingface.co/docs/trl
- alpaca-zh Dataset. https://huggingface.co/datasets/shibing624/alpaca-zh

---

## License

This project is for educational purposes.
