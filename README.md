# Qwen2.5 LoRA Instruction Finetuning

> 一份完整的大模型指令微调学习日志。从选模型、定方案、跑训练到做评估，每一步都记录了**为什么这么选**而非仅仅**做了什么**。
>
> 全程在 Google Colab 免费 T4 GPU 上完成，总成本 ¥0。

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/lhh737/Qwen-Lora-finetune/blob/main/finetune_qwen.ipynb)

---

## 写在前面

这个项目不是为了复现某个论文，也不是为了做一个产品。目的是**亲自动手走一遍 LLM 微调的完整流程**，在实践中理解每一个环节的技术选型背后的 trade-off。

如果你也在学大模型微调，希望这份记录能帮你少走一些弯路。

---

## 目录

- [1. 模型选型：为什么是 Qwen2.5？](#1-模型选型为什么是-qwen25)
- [2. 微调方案：为什么是 LoRA 不是全量微调？](#2-微调方案为什么是-lora-不是全量微调)
- [3. 数据准备：为什么是指令数据？](#3-数据准备为什么是指令数据)
- [4. LoRA 配置：rank 怎么选？target_modules 是什么？](#4-lora-配置rank-怎么选target_modules-是什么)
- [5. 训练过程：loss 降到多少算正常？](#5-训练过程loss-降到多少算正常)
- [6. 效果评估：怎么证明模型变好了？](#6-效果评估怎么证明模型变好了)
- [7. 模型量化：为什么部署需要量化？](#7-模型量化为什么部署需要量化)
- [8. 总结与反思](#8-总结与反思)

---

## 1. 模型选型：为什么是 Qwen2.5？

2025-2026 年可选的开源 LLM 大致分几类：

| 模型 | 中文能力 | 生态成熟度 | 硬件需求 |
|------|---------|-----------|---------|
| Qwen2.5 (阿里) | ★★★★ | ★★★★ | 适中 |
| LLaMA 3 (Meta) | ★★★ | ★★★★★ | 适中 |
| DeepSeek-V2/Lite | ★★★★ | ★★★ | 较低 |
| ChatGLM-4 (智谱) | ★★★★★ | ★★★ | 较高 |
| Yi (零一) | ★★★★ | ★★★ | 适中 |

选择 Qwen2.5 的理由：

1. **中文原生支持最好**。LLaMA 虽然生态最完善，但它是英文为主的模型，做中文任务需要额外的词表扩展和中文语料继续训练，工作量多一层。Qwen2.5 从预训练阶段就包含了大量中文数据。

2. **HuggingFace 生态兼容**。Qwen2.5 完全兼容 Transformers 库的接口，model_id 可以直接传给 `AutoModelForCausalLM.from_pretrained()`，不需要改代码。

3. **模型系列完整**。从 0.5B、1.5B、7B、14B 到 72B 都有。在 Colab T4（16GB）上 1.5B 可以轻松跑全精度 LoRA，7B 可以用 QLoRA（4-bit 加载）跑。同一系列不同规模，方便实验对比。

**实际选择**：先用 1.5B 快速验证流程（~8 分钟），再试 7B 看更大模型的收益（~40 分钟）。

---

## 2. 微调方案：为什么是 LoRA 不是全量微调？

### 全量微调（Full Fine-tuning）

- 所有参数都更新
- 7B 模型需要约 56GB 显存（BF16）+ 优化器状态 ~112GB → **总共 ~168GB**
- 至少需要 4-8 张 A100
- 对于个人学习和中小团队，基本不可行

### LoRA（Low-Rank Adaptation）

LoRA 的核心思想：**冻结原始权重，只训练两个低秩矩阵**。

```
原始: W ∈ ℝ^(d×d)         # 冻结，不更新
LoRA: W + BA              # 只训练 B ∈ ℝ^(d×r), A ∈ ℝ^(r×d)
r << d  (通常 r=8/16/32)
```

- 7B 模型用 rank=16 LoRA，可训练参数量只有 ~0.1%
- 训练时显存从 ~168GB 降到 ~12GB（QLoRA 甚至可降到 ~6GB）
- **单张 24GB 显卡就能跑**

### QLoRA 的进一步优化（7B 场景）

QLoRA = 4-bit 量化加载 + LoRA。基座模型先用 4-bit 加载（NF4 格式），显存降到原来的约 1/4，再在上面加 LoRA。

**代价**：训练速度比 FP16 LoRA 慢约 20-30%。

**选择结论**：1.5B 用 LoRA（全精度），7B 用 QLoRA（4-bit 加载）。这个项目以 1.5B LoRA 为主流程。

---

## 3. 数据准备：为什么是指令数据？

### 指令微调 vs 继续预训练

| | 继续预训练 | 指令微调（SFT） |
|---|---|---|
| 目标 | 让模型学更多知识 | 让模型学会遵循指令 |
| 数据格式 | 纯文本 | (instruction, output) 对 |
| 效果 | 增加知识量但不一定更听话 | 提升交互质量和对话能力 |
| 数据量需求 | 大（数十亿 token） | 小（几千到几万条） |

我们的目标是让模型**更好地回答技术问题**，所以用指令微调。数据选择了开源的中文指令数据集 alpaca-zh（基于斯坦福 Alpaca 翻译 + 清洗）。

### Chat Template 的作用

不同模型的对话格式不一样：

- Qwen2.5: `<|im_start|>user\n...<|im_end|>\n<|im_start|>assistant\n...<|im_end|>`
- LLaMA 3: `<|begin_of_text|><|start_header_id|>user<|end_header_id|>...<|eot_id|>`
- ChatGLM: `[GM]...`

**如果不统一格式，模型不知道哪里是问题哪里是回答，训练会无效。** 所以第一步就是把所有数据转成目标模型认识的格式。

---

## 4. LoRA 配置：rank 怎么选？target_modules 是什么？

### Rank 的选择

LoRA 的 rank（r）决定了低秩矩阵的维度：

- **r 越小**（如 4、8）：参数量少，训练快，但表达能力有限
- **r 越大**（如 32、64）：表达能力强，但参数量大，接近全量微调

社区经验：**r=16 是大多数场景的 sweet spot**。

为什么不是越大越好？

> 实验表明（参考 QLoRA 论文），r 从 8 到 16 有明显的效果提升，但从 16 到 32 提升边际递减。因为 LoRA 本身是在**低秩子空间**中做适配，而语言模型的权重变化量本身就是低秩的。r 超过这个本质秩之后，增加再多也只是在学噪声。

### target_modules 怎么选？

LoRA 可以加到 Transformer 的不同层：

- **Attention 层**（q_proj, k_proj, v_proj, o_proj）：最常用，直接影响信息交互
- **MLP 层**（gate_proj, up_proj, down_proj）：影响特征变换

大多数实践只微调 Attention 层，因为：
1. 效果已经足够好
2. 参数量少一半，训练更快
3. 减少过拟合风险

本项目的配置：

```python
LoraConfig(
    r=16,                    # 社区验证的稳妥选择
    lora_alpha=32,           # alpha = 2*r，控制更新幅度
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.05,       # 防止过拟合
    bias="none",
    task_type="CAUSAL_LM",   # 因果语言模型（生成任务）
)
```

---

## 5. 训练过程：loss 降到多少算正常？

### Loss 怎么看

训练过程中每 10 步打印一次 loss。loss 是**交叉熵损失**，数值越低表示模型的预测越准确。

经验参考（基于 Qwen2.5-1.5B + 3000 条数据）：

| 阶段 | Loss 范围 | 说明 |
|------|----------|------|
| 起始 | ~1.8-2.0 | 模型刚开始学，预测接近随机 |
| 训练中 | 1.0 → 0.6 | 稳步下降，说明学到模式 |
| 结束 | ~0.5-0.7 | 收敛，继续训练收益不大 |

**注意事项**：

- loss 在 0.5 以下可能表示**过拟合**（背下了训练数据而不是泛化）
- 如果 loss 不降，检查：学习率、数据格式、是否用对了 Chat Template
- 1.5B 模型 3000 条数据 2 epoch 大约 8 分钟

### 训练资源占用

| 配置 | 显存占用 | 训练时间 |
|------|---------|---------|
| 1.5B LoRA (FP16) | ~6GB | ~8 min |
| 7B QLoRA (4-bit) | ~12GB | ~40 min |

---

## 6. 效果评估：怎么证明模型变好了？

这是整个项目中最能体现**工程判断力**的部分。

### 为什么不依赖标准指标？

BLEU、ROUGE 等指标的问题：

- BLEU 基于 n-gram 精确匹配，"你叫什么名字"和"请问你的名字是"这种语义等价但字面不同的回答会得低分
- 对于 LLM 这种生成式模型，字面匹配不能反映回答质量

### 我用的方法：关键词命中率

设计 20 道技术问答（覆盖 Transformer、Python、Docker、模型量化等方向），每道题预定义 3-4 个核心关键词。

```
题目："LoRA 微调的核心原理是什么？"
关键词：["低秩", "冻结", "插入", "参数量"]

回答 A："通过低秩分解..." → 命中 3/4 = 75%
回答 B："这是一种微调方法..." → 命中 0/4 = 0%
```

**这个方法的优点**：
- 简单、可复现
- 聚焦在"关键概念是否覆盖"而不是"措辞是否一致"
- 容易解释给面试官

**局限性**（我也清楚）：
- 关键词覆盖不全，模型可能用不同的词表达同样的意思
- 无法评判回答的深度和逻辑性
- 所以结合了人工阅读部分回答做定性判断

### 运行结果

```
评估汇总 (20 题)
基座模型平均命中率:  38%
微调后平均命中率:  65%
相对提升:          +71%
微调更好/更差:      15/3
```

> **诚实地说**：1.5B 模型的回答仍然不够好，很多回答比较简短。但如果换成 7B 模型，效果会有明显提升。技术流程是一样的，模型规模只是算力问题。

---

## 7. 模型量化：为什么部署需要量化？

### 量化的动机

一个 FP16 的 7B 模型需要 **14GB** 的显存才能加载。而一张消费级显卡（RTX 3090）只有 24GB。

量化就是把模型权重从高精度（FP16, 2 bytes）压缩到低精度（INT4, 0.5 bytes）：

```
FP16:  每个参数 2 bytes → 7B × 2 = 14GB
INT4:  每个参数 0.5 bytes → 7B × 0.5 = 3.5GB
```

### AWQ vs GPTQ

| | AWQ | GPTQ |
|---|---|---|
| 原理 | 根据激活值分布保护重要权重 | 基于二阶优化的逐层量化 |
| 精度损失 | ~0.5-1% | ~0.5-1% |
| 推理速度 | 略快 | 中等 |
| 生态支持 | vLLM 原生支持 | 较为成熟 |

两者效果接近，这个项目使用 GPTQ 量化为 INT4。

**量化后的精度损失在可接受范围内**，但模型体积缩小 60%，推理速度提升约 2-3 倍。这是实际部署中非常常见的 trade-off。

---

## 8. 总结与反思

### 我学到了什么

1. **LoRA 不是万能的**。它擅长"教会模型回答格式"，但不擅长"给模型注入新知识"。如果想让模型知道它原本不知道的事实，需要用 RAG 或者继续预训练。

2. **评估比训练更难**。模型训练有 loss 作为标准反馈，但效果好坏需要自己设计评测方式。评测设计的好坏直接决定了你能否正确判断模型是否变好了。

3. **资源限制决定了技术选型**。如果不是只有一张 T4，我可能会用更大的模型、更多的数据、更长的训练。但正是这种限制，让我深入理解了 LoRA 和量化的必要性。

### 如果重新来/继续做

- [ ] 试 7B QLoRA，看更大模型的效果提升
- [ ] 在更多样化的领域数据上微调（不只是技术问答）
- [ ] 用 LLM-as-Judge 做自动化评估，覆盖面更广
- [ ] 部署到 HuggingFace Spaces 做一个 Demo
- [ ] 对比 LoRA vs QLoRA 的效果和速度差异

---

## 项目结构

```
qwen-lora-finetune/
├── README.md               ← 你正在看的这份学习日志
├── finetune_qwen.ipynb     ← Colab 笔记本（全部流程）
├── data_prep.py              数据准备
├── train_lora.py              LoRA 训练
├── evaluate.py                效果评估
├── quantize.py                量化导出
└── requirements.txt
```

## 快速开始

```bash
# 在本地或 Colab 中运行
git clone https://github.com/lhh737/Qwen-Lora-finetune.git
cd Qwen-Lora-finetune

# 上传 finetune_qwen.ipynb 到 Google Colab，选 T4 GPU，逐格运行
```

或直接点击：[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/lhh737/Qwen-Lora-finetune/blob/main/finetune_qwen.ipynb)

## 技术栈

Python · PyTorch · HuggingFace Transformers · PEFT · TRL · LoRA · GPTQ/AWQ · Google Colab

## 参考

- [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685)
- [QLoRA: Efficient Finetuning of Quantized LLMs](https://arxiv.org/abs/2305.14314)
- [PEFT Documentation](https://huggingface.co/docs/peft)
- [Qwen2.5 Technical Report](https://arxiv.org/abs/2412.15115)
- [alpaca-zh Dataset](https://huggingface.co/datasets/shibing624/alpaca-zh)
