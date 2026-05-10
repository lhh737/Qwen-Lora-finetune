"""
数据准备：基于 alpaca-zh 构建训练数据 + 自建 100 条技术问答测试集
"""
import json, os
from datasets import load_dataset

os.makedirs("./data", exist_ok=True)

# ===== 1. 加载开源指令数据（训练用） =====
print("[1/3] 加载 alpaca-zh 数据集...")
dataset = load_dataset("shibing624/alpaca-zh", split="train")
dataset = dataset.select(range(5000))

def format_chat(example):
    """统一为 ChatML 格式"""
    return {
        "text": (
            f"<|im_start|>user\n{example['instruction']}\n"
            f"<|im_end|>\n"
            f"<|im_start|>assistant\n{example['output']}\n<|im_end|>"
        )
    }

dataset = dataset.map(format_chat)
dataset.save_to_disk("./data/alpaca_5k")
print(f"  → 训练数据: {len(dataset)} 条, 已保存至 ./data/alpaca_5k")

# ===== 2. 自建测试集（评估用） =====
print("[2/3] 构建技术问答测试集...")
test_set = [
    # ── Transformer / LLM 基础 ──
    {"question": "Transformer 中的多头注意力机制是如何工作的？",
     "reference": "将输入分别映射到多个头，每个头独立计算注意力，最后拼接并线性变换。"},
    {"question": "什么是位置编码？Transformer 中为什么需要它？",
     "reference": "位置编码用于给模型提供序列中 token 的位置信息，因为自注意力本身不具备位置感知能力。"},
    {"question": "GPT 系列模型和 BERT 在架构上有什么主要区别？",
     "reference": "GPT 使用单向自注意力（从左到右），BERT 使用双向自注意力。GPT 是自回归生成模型，BERT 是编码器模型。"},
    {"question": "什么是 KV Cache？在推理中起到什么作用？",
     "reference": "KV Cache 缓存历史 token 的 Key 和 Value，避免每步重复计算，加速自回归推理。"},
    {"question": "LoRA 微调的核心原理是什么？为什么它比全量微调更高效？",
     "reference": "LoRA 冻结预训练权重，在权重矩阵旁插入低秩分解矩阵，只训练低秩矩阵，参数量远少于全量微调。"},
    {"question": "什么是 RAG？它主要解决什么问题？",
     "reference": "RAG 是检索增强生成，在生成前从外部知识库检索相关文档作为上下文，解决模型知识不足和幻觉问题。"},
    {"question": "ReAct 推理范式是什么？它和单纯的 CoT 有什么区别？",
     "reference": "ReAct 交替进行推理和行动，思考后执行工具调用，根据结果继续推理。CoT 只推理不行动。"},
    {"question": "AWQ 量化和 GPTQ 量化有什么不同？",
     "reference": "AWQ 根据激活值分布决定缩放因子保护重要权重，GPTQ 基于二阶近似逐层量化，两者都是 INT4 权重量化。"},
    {"question": "什么是 Plan-and-Execute 范式？",
     "reference": "先由规划模块将任务拆分为子任务，再由执行模块按序执行，适用于复杂多步骤任务。"},
    {"question": "什么是 PagedAttention？它解决了什么问题？",
     "reference": "PagedAttention 将 KV Cache 分页管理，类似操作系统的虚拟内存，解决显存碎片和浪费问题。"},

    # ── Python / 工程 ──
    {"question": "Python 中装饰器是什么？写一个简单的例子。",
     "reference": "装饰器是高阶函数，接受函数作为参数并返回增强版函数。"},
    {"question": "什么是 Python 生成器？yield 关键字如何工作？",
     "reference": "生成器通过 yield 每次返回一个值并暂停执行，下次调用从暂停处继续，节省内存。"},
    {"question": "什么是 GIL？它有什么影响？",
     "reference": "全局解释器锁，限制同一时刻只有一个线程执行 CPython 字节码，影响多线程 CPU 密集任务。"},
    {"question": "FastAPI 和 Flask 的主要区别是什么？",
     "reference": "FastAPI 基于异步，支持自动生成 OpenAPI 文档和请求校验，性能高于 Flask。"},
    {"question": "什么是上下文管理器？with 语句是如何工作的？",
     "reference": "上下文管理器定义进入和退出时的行为，通过 __enter__ 和 __exit__ 实现，with 自动调用。"},
    {"question": "Docker 中 COPY 和 ADD 指令有什么区别？",
     "reference": "COPY 仅复制文件，ADD 额外支持 URL 下载和自动解压压缩包，推荐优先用 COPY。"},
    {"question": "PostgreSQL 中索引的类型有哪些？",
     "reference": "B-tree, Hash, GiST, GIN, BRIN 等，B-tree 最常用，适合等值和范围查询。"},
    {"question": "Redis 的过期策略有哪些？",
     "reference": "定时删除、惰性删除、定期删除。Redis 实际使用惰性+定期组合策略。"},
    {"question": "Git 中 rebase 和 merge 的主要区别是什么？",
     "reference": "merge 创建新的合并提交保留分支历史，rebase 将提交线性重放使历史更清晰。"},
    {"question": "HTTP 中 GET 和 POST 的本质区别是什么？",
     "reference": "GET 是幂等的用于获取资源，参数在 URL 中；POST 非幂等用于提交数据，参数在请求体中。"},

    # ── LLM 应用 ──
    {"question": "LangChain 中的 Chain 和 Agent 有什么区别？",
     "reference": "Chain 是固定执行流程，Agent 根据输入动态决定执行步骤并调用工具。"},
    {"question": "什么是 Prompt Engineering？有哪些常见技巧？",
     "reference": "设计输入提示引导模型输出。常见技巧：Few-shot、CoT、角色设定、输出格式约束。"},
    {"question": "LLM 推理时 temperature 参数的作用是什么？",
     "reference": "控制输出随机性，temperature 越高输出越多样，越低越确定，0 时贪婪解码。"},
    {"question": "什么是 top-p 采样？",
     "reference": "从累积概率超过 p 的最小 token 集合中采样，又称核采样，动态调整候选集大小。"},
    {"question": "什么是模型幻觉？如何缓解？",
     "reference": "模型生成看似合理但实际错误的内容。缓解方法：RAG、微调、约束解码、多次采样。"},
    {"question": "什么是 LLM-as-Judge？",
     "reference": "用一个 LLM 作为评估者，对另一个 LLM 的输出质量进行打分或比较，常用于评估。"},
    {"question": "什么是 RAGAS 框架？主要评估哪些维度？",
     "reference": "RAGAS 是 RAG 评估框架，评估忠实度、答案相关性、上下文精度和召回率。"},
    {"question": "Multi-Agent 系统相比单 Agent 有什么优势？",
     "reference": "分工协作：不同 Agent 负责不同子任务，降低单 Agent 复杂度，提升鲁棒性和模块化。"},
    {"question": "什么是 Tool Calling？LLM 如何决定调用哪个工具？",
     "reference": "LLM 输出结构化 JSON 表示工具调用意图，系统解析后执行并返回结果。"},
    {"question": "SFT 和 RLHF 的区别是什么？",
     "reference": "SFT 是监督微调，用人工标注数据训练；RLHF 基于人类反馈强化学习，进一步对齐偏好。"},

    # ── 深度学习基础 ──
    {"question": "Batch Normalization 和 Layer Normalization 的区别？",
     "reference": "BN 对 batch 维度归一化，依赖 batch size；LN 对特征维度归一化，不依赖 batch，适合 NLP。"},
    {"question": "什么是梯度消失和梯度爆炸？如何缓解？",
     "reference": "深层网络中梯度指数级衰减或增长。缓解：残差连接、梯度裁剪、归一化、合适的激活函数。"},
    {"question": "Dropout 的原理和作用是什么？",
     "reference": "训练时随机丢弃部分神经元，防止过拟合，相当于集成学习。"},
    {"question": "什么是学习率预热？为什么需要它？",
     "reference": "训练初期从小的学习率逐步增大到目标值，避免初始大梯度导致不稳定。"},
    {"question": "Adam 优化器和 SGD 相比有什么优势？",
     "reference": "Adam 自适应调整每个参数的学习率，结合动量和 RMSProp，收敛更快更稳定。"},
    {"question": "交叉熵损失函数在处理分类问题时如何工作？",
     "reference": "衡量预测分布和真实分布的差异，值越小预测越接近真实标签。"},
    {"question": "什么是过拟合？如何判断和防止？",
     "reference": "模型在训练集上表现好但测试集差。判断：训练/验证 loss 差距大。防止：正则化、数据增强、早停。"},
    {"question": "什么是注意力机制中的 Q、K、V？",
     "reference": "Query 查询当前 token，Key 被查询的 token 标识，Value 实际信息内容，Attention 是 QK 相似度加权 V。"},
    {"question": "什么是残差连接？为什么能训练深层网络？",
     "reference": "将输入直接加到输出上，梯度可以绕过中间层直接传播，缓解梯度消失。"},
    {"question": "PyTorch 中 dataloader 的 num_workers 参数作用是什么？",
     "reference": "控制数据加载的子进程数，多进程并行加载数据，避免 GPU 等待数据加载。"},

    # ── 量化 / 推理 / 部署 ──
    {"question": "模型量化为什么可以减少显存占用？",
     "reference": "将 FP32/FP16 的权重用 INT8/INT4 存储，每个参数占用的比特数减少，显存占用相应减少。"},
    {"question": "INT4 量化相比 FP16 推理速度会提升吗？为什么？",
     "reference": "取决于硬件。计算瓶颈时更少的字节搬运提升速度，但需要硬件支持 INT4 计算加速。"},
    {"question": "什么是连续批处理？和传统批处理有什么不同？",
     "reference": "连续批处理在序列级别动态增删请求，而非等整个 batch 都生成完，提高 GPU 利用率。"},
    {"question": "vLLM 比 HuggingFace 原生推理快多少？为什么？",
     "reference": "vLLM 使用 PagedAttention + 连续批处理，通常比原生推理快 10-20 倍。"},
    {"question": "什么是 speculative decoding？",
     "reference": "用小模型快速草稿，大模型并行验证草稿，加速推理同时保持大模型输出质量。"},
    {"question": "Docker 中如何让容器使用宿主机的 GPU？",
     "reference": "安装 NVIDIA Container Toolkit，运行容器时加 --gpus all 参数。"},
    {"question": "推理延迟 P50 和 P99 分别代表什么？为什么都要关注？",
     "reference": "P50 是中位数延迟，P99 是 99% 请求的延迟上限。P50 体感，P99 反映尾延迟问题。"},
    {"question": "什么是 GPU 显存碎片化？如何缓解？",
     "reference": "频繁申请释放显存导致碎片。缓解：预分配缓存、PagedAttention、vLLM 显存管理。"},
    {"question": "模型部署时 health check 的作用是什么？",
     "reference": "定期检查服务是否正常响应，负载均衡器根据 health check 结果决定是否分发流量。"},
    {"question": "API 服务中 rate limit 为什么重要？",
     "reference": "防止单个客户端过度占用资源导致服务不可用，保证服务质量公平分配。"},

    # ── 大模型生态 ──
    {"question": "什么是 Chat Template？为什么需要它？",
     "reference": "不同模型有不同的对话格式封装规则，Chat Template 统一管理，避免格式错误。"},
    {"question": "HuggingFace Model Hub 上可以上传哪些内容？",
     "reference": "模型权重、分词器、配置文件、模型卡片、示例代码，支持版本管理和协作。"},
    {"question": "模型参数量、训练数据量和模型性能的关系是什么？",
     "reference": "通常参数量越大能力越强，但需要足够高质量数据配合，存在 Scaling Law 指导。"},
    {"question": "什么是 embedding 模型？它和 LLM 有什么关系？",
     "reference": "embedding 模型将文本映射为向量，LLM 用这些向量作为输入或检索。RAG 中两者配合使用。"},
    {"question": "Fine-tuning 和 In-Context Learning 有什么区别？",
     "reference": "Fine-tuning 更新模型权重，ICL 在推理时提供示例不更新权重。ICL 灵活但效果不如 Fine-tuning 稳定。"},
    {"question": "什么是 P-tuning / Prefix Tuning？",
     "reference": "在输入层或每层插入可学习的 prompt 向量，只训练这些向量不训练原模型。"},
    {"question": "什么是 Grouped Query Attention？它解决了什么问题？",
     "reference": "多组 Query 共享一组 Key-Value，减少 KV Cache 大小，降低推理显存需求。"},
    {"question": "什么是 MoE（混合专家）？",
     "reference": "模型由多个专家子网络组成，路由网络选择激活哪些专家，同等计算量下用更多参数。"},
    {"question": "什么是 RLHF 中的 Reward Model？",
     "reference": "学习人类偏好给模型输出打分的模型，用于指导强化学习阶段优化策略模型。"},
    {"question": "DPO 和 RLHF 有什么不同？",
     "reference": "DPO 直接基于偏好对优化策略，不需要单独训练 Reward Model，训练更简单稳定。"},
]
print(f"  → 测试集: {len(test_set)} 条")

with open("./data/test_100.json", "w", encoding="utf-8") as f:
    json.dump(test_set, f, ensure_ascii=False, indent=2)

# ===== 3. 摘录 =====
print("[3/3] 打印示例数据...")
print(f"\n训练数据示例:\n{dataset[0]['text'][:200]}...\n")
print(f"测试集示例: {test_set[0]['question']}")

print("\n✓ 数据准备完成")
