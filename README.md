# Qwen2.5 LoRA 指令微调

基于 Qwen2.5 的 LoRA 指令微调完整流程，涵盖数据准备、训练、评估和量化。可在 Google Colab 免费 T4 GPU 上运行。

[![在 Colab 中打开](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/lhh737/Qwen-Lora-finetune/blob/main/finetune_qwen.ipynb)

---

## 目录

- [概述](#概述)
- [流程概览](#流程概览)
- [环境配置](#环境配置)
- [模型选型](#模型选型)
- [微调方法](#微调方法)
- [数据准备](#数据准备)
- [LoRA 配置](#lora-配置)
- [训练](#训练)
- [评估](#评估)
- [量化](#量化)
- [项目结构](#项目结构)
- [运行结果](#运行结果)
- [生产环境差距分析](#生产环境差距分析)
- [参考资料](#参考资料)

---

## 概述

**问题**：通用大模型在垂直领域表现不佳，缺乏专业术语知识和特定领域的回答规范。

**方案**：使用 LoRA（低秩适配）进行参数高效微调。仅训练约 0.1% 的参数，即可在消费级 GPU 上完成领域适配。

**范围**：本项目覆盖完整流程：
1. 数据获取与格式化
2. 模型与方法选型
3. LoRA 微调训练
4. 定量与定性评估
5. 模型量化导出

**约束条件**：零成本（Google Colab 免费版），单张 T4 GPU（16GB 显存），3000 条训练样本。

---

## 流程概览

```
数据准备              训练                    评估                    量化
─────────────────    ──────────              ───────────              ─────────────
alpaca-zh 数据集      Qwen2.5 基座模型       基座 vs LoRA 输出        合并 LoRA 权重
  ↓                     ↓                       ↓                       ↓
ChatML 格式转换        LoRA 适配器注入          20 道 QA 测试集          GPTQ/AWQ INT4
  ↓                     ↓                       ↓                       ↓
3000 条样本            SFTTrainer 训练          关键词命中率评估         量化模型导出
                      (2 epoch, ~8 分钟)                                 (体积缩小 ~60%)
```

---

## 环境配置

### 平台对比

| 平台 | GPU | 显存 | 费用 | 时长限制 | 适用场景 |
|------|-----|------|------|---------|---------|
| **Google Colab（免费）** | T4 | 16 GB | ¥0 | ~1 小时 | 本项目 |
| Google Colab（Pro） | T4/V100/A100 | 16-40 GB | ~¥80/月 | ~24 小时 | 更大模型 |
| Kaggle Notebooks | T4/P100 | 16 GB | ¥0 | 30 小时/周 | 备选 |
| AutoDL | 3090/4090/A100 | 24-80 GB | ¥2-15/时 | 无 | 生产级训练 |
| 本地 GPU | RTX 3060-4090 | 6-24 GB | 已有硬件 | 无 | 小型实验 |

### 快速开始

```bash
# 1. 克隆仓库
git clone https://github.com/lhh737/Qwen-Lora-finetune.git
cd Qwen-Lora-finetune

# 2. 在 Colab 中打开
#    - 访问 https://colab.research.google.com
#    - 上传 finetune_qwen.ipynb
#    - 运行时 → 更改运行时类型 → T4 GPU
#    - 逐格运行 (Shift+Enter)
```

各模型预估运行时间：

| 模型 | 方法 | 预计耗时 |
|------|------|---------|
| Qwen2.5-1.5B | LoRA (FP16) | ~8 分钟 |
| Qwen2.5-7B | QLoRA (4-bit) | ~40 分钟 |

---

## 模型选型

### 开源 LLM 对比（2025-2026）

| 模型 | 出品方 | 中文能力 | 英文能力 | 代码能力 | 生态成熟度 |
|------|-------|---------|---------|---------|-----------|
| **Qwen2.5** | 阿里巴巴 | ★★★★☆ | ★★★★☆ | ★★★★☆ | ★★★★☆ |
| LLaMA 3.1 | Meta | ★★★☆☆ | ★★★★★ | ★★★★☆ | ★★★★★ |
| DeepSeek-V3/R1 | 深度求索 | ★★★★★ | ★★★★★ | ★★★★★ | ★★★☆☆ |
| GLM-4 | 智谱 AI | ★★★★★ | ★★★☆☆ | ★★★☆☆ | ★★★☆☆ |
| Yi 1.5 | 零一万物 | ★★★★☆ | ★★★★☆ | ★★★☆☆ | ★★★☆☆ |

### 选型依据

选择 Qwen2.5 的三个关键因素：

**1. 中文 Token 效率**

Qwen2.5 词表大小为 151,642，包含大量中文字符。中文文本经 Qwen2.5 分词后 token 数量显著低于 LLaMA 3（后者主要针对英文优化），直接提升训练和推理效率。

**2. HuggingFace 生态兼容**

Qwen2.5 遵循标准 Transformers 接口：

```python
# 所有尺寸的模型统一加载方式
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-1.5B",     # 修改此 ID 即可切换模型
    torch_dtype="auto",
    device_map="auto",
    trust_remote_code=True,
)
```

与 PEFT、TRL、vLLM 等工具完全兼容，无需适配代码。

**3. 完整模型谱系**

同一架构覆盖 0.5B 至 72B，可在小模型上快速验证后放大。从 1.5B 切换到 7B 只需修改 `MODEL_ID`。

### 推荐配置

| 场景 | 模型 | 方法 | 显存需求 |
|------|------|------|---------|
| 首次运行、快速迭代 | Qwen2.5-1.5B | LoRA (FP16) | ~6 GB |
| 更好回答质量 | Qwen2.5-7B | QLoRA (NF4) | ~12 GB |

两种配置的技术流程完全一致，差异仅在于训练速度和输出质量。

---

## 微调方法

### 方案对比

| 方法 | 可训练参数量 | 7B 模型显存 | 速度 | 效果 |
|------|------------|------------|------|------|
| 全量微调 | 100% (7B) | ~168 GB | 慢 | 最优 |
| LoRA (r=16) | ~0.1% (7M) | ~14 GB | 快 | 接近全量微调 |
| QLoRA (NF4 + LoRA) | ~0.1% (7M) | ~6 GB | 中等 | 接近 LoRA |

### 全量微调显存分析

7B 模型 FP16 全量微调所需显存：

| 组件 | 显存 | 计算方式 |
|------|------|---------|
| 模型权重 | 14 GB | 7B × 2 bytes (FP16) |
| 梯度 | 14 GB | 7B × 2 bytes (FP16) |
| Adam 优化器状态 | 56 GB | 7B × 8 bytes (FP32 动量 + 方差) |
| 小计（不含激活值） | 84 GB | |
| 激活值（每 batch） | 20-80 GB | 取决于模型和 batch size |
| **总计** | **~110-168 GB** | |

全量微调需要 4-8 张 A100 GPU（每张 80GB），个人学习场景不可行。LoRA 将可训练参数量减少至约 0.1%，显存需求降低 10-20 倍。

### LoRA 原理

LoRA（Low-Rank Adaptation）将权重更新约束为低秩分解形式。对于预训练权重矩阵 _W₀_ ∈ ℝ^(d×k)：

_W = W₀ + ΔW = W₀ + BA_，其中 _B_ ∈ ℝ^(d×r)，_A_ ∈ ℝ^(r×k)，_r_ ≪ min(d, k)

训练时 _W₀_ 冻结，仅更新 _A_ 和 _B_。前向传播变为：

_h = Wx = W₀x + BAx_

可训练参数量从 _d²_ 降至 _2dr_。对于 d=4096、r=16 的典型层，参数量减少 99.2%。

LoRA 有效性的理论基础：预训练模型的权重更新量具有较低的**本质秩**（intrinsic rank），任务特定的有效更新维度远小于参数量空间。

### QLoRA

QLoRA（Quantized LoRA）将基座模型以 4-bit NF4 格式加载后再应用 LoRA，7B 模型显存占用从 14GB 降至约 3.5GB。代价是训练速度降低 20-30%（每次前向传播需反量化）。

---

## 数据准备

### 指令微调 vs 继续预训练

| 维度 | 指令微调 (SFT) | 继续预训练 |
|------|---------------|-----------|
| 目标 | 学习回答格式和行为规范 | 注入领域知识 |
| 数据格式 | (指令, 回答) 配对 | 纯文本语料 |
| 数据量 | 数千至数万条 | 数亿至数十亿 token |
| 效果 | 提升指令遵循能力 | 提升知识覆盖 |

### 数据集对比

| 数据集 | 规模 | 质量 | 格式 | 获取方式 |
|-------|------|------|------|---------|
| **alpaca-zh** | 52K | ★★★☆ | 标准 | HuggingFace |
| BELLE | 3.5M | ★★★★ | 标准 | HuggingFace |
| Firefly | 1.1M | ★★★★ | 标准 | HuggingFace |

选择 alpaca-zh 的原因：HuggingFace 直接加载、格式规范统一、适合学习场景。项目使用 3000 条子集即可完成流程验证。

### ChatML 格式

不同模型使用不同的对话格式，必须与预训练格式一致：

```
Qwen2.5 (ChatML):
<|im_start|>user
{问题}
<|im_end|>
<|im_start|>assistant
{回答}
<|im_end|>

LLaMA 3:
<|begin_of_text|><|start_header_id|>user<|end_header_id|>
{问题}<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>
{回答}<|eot_id|>
```

格式不匹配会导致训练失败（loss 不下降）或模型输出不可用。

---

## LoRA 配置

### 参数说明

**rank (r)**：控制低秩矩阵的维度。rank 越高表达能力越强，但参数量也越大。

| 配置 | 参数量 | 适用场景 |
|------|--------|---------|
| r=8 | 2×d×8 | 最小有效配置 |
| **r=16** | 2×d×16 | **推荐起始值** |
| r=32 | 2×d×32 | 高容量需求 |

实验表明 r=16 在大多数任务中达到收益平衡点，继续增大 rank 的边际收益递减。

**lora_alpha**：LoRA 更新的缩放因子。实际更新量为 `(lora_alpha / r) × BA`。`lora_alpha = 2r`（缩放因子 2）是最常见配置。

**target_modules**：LoRA 应用的目标参数矩阵。通常选择全部注意力层（q_proj, k_proj, v_proj, o_proj），在表达能力和参数量之间取得平衡。

**lora_dropout**：LoRA 层的 dropout 比率。小数据集（数千条）训练时 0.05 提供有效的正则化。

### 默认配置

```python
LoraConfig(
    r=16,
    lora_alpha=32,                      # 2r，缩放因子 2
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",                        # 不训练偏置项
    task_type="CAUSAL_LM",              # 因果语言模型
)
```

---

## 训练

### 关键参数

| 参数 | 值 | 说明 |
|------|------|------|
| `per_device_train_batch_size` | 4 | 每张 GPU 每步样本数 |
| `gradient_accumulation_steps` | 4 | 梯度累积步数，等效 batch = 4×4=16 |
| `num_train_epochs` | 2 | 训练轮数 |
| `learning_rate` | 2e-4 | 学习率，LoRA 通常比全量微调高 2-4 倍 |
| `fp16` | True | 混合精度训练，节省显存 |

### Loss 指标参考

Qwen2.5-1.5B + 3000 条数据 + 2 epoch 的预期 loss 曲线：

| Step | Loss | 说明 |
|------|------|------|
| 0 | ~1.8 | 初始状态 |
| 20 | ~1.2 | 快速下降，学习对话格式 |
| 50 | ~0.9 | 稳定下降 |
| 100 | ~0.7 | 接近收敛 |
| 结束 | ~0.5-0.6 | 训练完成 |

### 常见问题

| 现象 | 原因 | 解决方案 |
|------|------|---------|
| Loss 不降（>2.0） | 数据格式错误 | 检查 ChatML 格式 |
| Loss < 0.1 | 过拟合 | 减少 epoch、增加数据量 |
| Loss 振荡 | 学习率过高 | 降至 1e-4 或 5e-5 |
| CUDA OOM | Batch size 过大 | 降低 per_device_train_batch_size |

---

## 评估

### 评估方法

生成式模型的评估不同于分类任务。BLEU 和 ROUGE 等指标基于 n-gram 字面匹配，无法反映语义质量。

本项目使用**关键词命中率**方法：为每道测试题预定义 3-4 个核心关键词，计算回答中包含的关键词比例。

| 测试题 | 关键词 |
|--------|--------|
| LoRA 微调的核心原理是什么？ | 低秩, 冻结, 插入, 参数量 |
| 什么是 KV Cache？ | 缓存, Key, Value, 加速 |

**评分公式**：`命中率 = 回答中出现的关键词数 / 总关键词数`

**优势**：透明、可复现、可诊断。**局限性**：无法评估同义词表达、概念使用正确性和回答深度。需结合人工阅读做定性判断。

### 预期结果

```
指标                   基座模型      微调后        变化
─────────────────────  ──────────    ──────────    ──────
平均命中率              ~38%          ~65%          +71%
回答更好的题数          —             15/20         —
回答更差的题数          —             3/20          —
```

---

## 量化

### 动机

模型精度与存储需求关系：

| 模型规模 | FP16 | INT4 | 压缩比 |
|---------|------|------|--------|
| 1.5B | 3 GB | 0.8 GB | 73% |
| 7B | 14 GB | 3.5 GB | 75% |

量化同时降低存储占用和推理延迟（减少内存与计算单元间的数据传输）。

### 方法对比

| 维度 | AWQ | GPTQ |
|------|-----|------|
| 原理 | 基于激活值分布保护重要权重 | 基于二阶 Hessian 优化 |
| 精度损失 | ~0.5% | ~0.5-1% |
| 生态 | vLLM 原生支持 | 工具链成熟 |

本项目通过 `quantize.py` 支持两种方法（默认 GPTQ），生产环境推荐使用 AWQ。

---

## 项目结构

```
qwen-lora-finetune/
├── README.md                   # 英文文档
├── README.zh.md                # 中文文档（本文件）
├── finetune_qwen.ipynb         # Colab 笔记本（主要入口）
├── data_prep.py                # 数据加载与格式化
├── train_lora.py               # LoRA 训练脚本（CLI，支持 argparse）
├── evaluate.py                 # 关键词命中率评估
├── quantize.py                 # 模型量化（GPTQ/AWQ）
├── requirements.txt            # Python 依赖
├── .gitignore
└── results/                    # 评估输出目录
```

---

## 运行结果

*待运行后补充。* 预期输出包括：
- 训练 loss 曲线
- 微调前后评估对比表
- 单题关键词命中率明细

---

## 生产环境差距分析

### 与工业部署的差距

| 维度 | 本项目 | 生产环境 |
|------|--------|---------|
| **数据** | 公开数据集，3K 条，通用领域 | 私有数据，数万至数十万条，垂直领域 |
| **训练** | 单卡 GPU，LoRA，2 epoch | 分布式训练，多阶段（SFT + RLHF/DPO） |
| **评估** | 20 题关键词测试 | 完整测试集，人工评估，A/B 测试 |
| **基础设施** | Colab 笔记本 | 容器化 API（vLLM/TGI），负载均衡，监控告警 |
| **模型管理** | 本地文件 | 模型注册中心，版本控制，CI/CD |
| **监控** | 无 | 延迟 P50/P99，吞吐量，错误率，数据漂移 |

### 后续学习方向

| 方向 | 具体行动 | 所需资源 |
|------|---------|---------|
| 规模扩展 | 使用 QLoRA 微调 7B 模型 | AutoDL 或 Colab Pro |
| 评估深化 | 实现 LLM-as-Judge 自动评估 | API 访问权限 |
| 生产部署 | vLLM + FastAPI + Docker | 本地 GPU 或云实例 |
| 原理深入 | 阅读 LoRA / QLoRA 论文原文 | 学术访问 |

---

## 参考资料

- Hu et al. "LoRA: Low-Rank Adaptation of Large Language Models." ICLR 2022.
- Dettmers et al. "QLoRA: Efficient Finetuning of Quantized LLMs." NeurIPS 2023.
- Qwen Team. "Qwen2.5 Technical Report." arXiv:2412.15115, 2024.
- HuggingFace PEFT 文档. https://huggingface.co/docs/peft
- HuggingFace TRL 文档. https://huggingface.co/docs/trl
- alpaca-zh 数据集. https://huggingface.co/datasets/shining624/alpaca-zh
