"""
数据准备脚本

作用：从 HuggingFace 加载开源中文指令数据集，统一格式为 ChatML，
      供 SFTTrainer 训练使用。

输出：
  - ./data/alpaca_5k/    格式化后的训练数据（HuggingFace Dataset 格式）
  - ./data/test_50.json   自建测试集（用于评估）

使用方法：
  python data_prep.py
"""

import json, os
from datasets import load_dataset

def main():
    os.makedirs("./data", exist_ok=True)

    # ── 1. 加载数据集 ──
    # shibing624/alpaca-zh 是基于斯坦福 Alpaca 翻译的中文指令数据集
    # 共 52K 条，我们取前 3000 条用于训练
    print("[1/3] 加载数据集...")
    dataset = load_dataset("shibing624/alpaca-zh", split="train")
    dataset = dataset.select(range(3000))
    print(f"  训练样本数: {len(dataset)}")

    # ── 2. 格式化数据 ──
    # Qwen2.5 使用 ChatML 格式：
    #   <|im_start|>user\n{问题}\n<|im_end|>\n<|im_start|>assistant\n{回答}\n<|im_end|>
    # SFTTrainer 需要数据集中有一个 "text" 字段，包含格式化后的完整文本
    print("[2/3] 格式化数据...")
    CHAT_TEMPLATE = """<|im_start|>user
{instruction}
<|im_end|>
<|im_start|>assistant
{output}
<|im_end|>"""

    def format_chat(ex):
        return {"text": CHAT_TEMPLATE.format(
            instruction=ex["instruction"],
            output=ex["output"]
        )}

    dataset = dataset.map(format_chat)
    dataset.save_to_disk("./data/alpaca_5k")
    print(f"  已保存到 ./data/alpaca_5k")

    # ── 3. 构建测试集 ──
    # 20 道技术问答，覆盖 LLM 基础、工程实践、深度学习理论
    # 每道题预定义了评分关键词
    print("[3/3] 构建测试集...")
    test_set = [
        {"question": "Transformer 中的多头注意力机制是如何工作的？",
         "keywords": ["多个头", "拼接", "线性变换", "子空间"]},
        {"question": "什么是 KV Cache？在推理中起到什么作用？",
         "keywords": ["缓存", "Key", "Value", "加速"]},
        {"question": "LoRA 微调的核心原理是什么？为什么比全量微调高效？",
         "keywords": ["低秩", "冻结", "插入", "参数量"]},
        {"question": "什么是 RAG？它主要解决什么问题？",
         "keywords": ["检索", "增强", "生成", "知识库"]},
        {"question": "ReAct 推理范式是什么？和 CoT 有什么区别？",
         "keywords": ["推理", "行动", "交替", "工具"]},
        {"question": "AWQ 量化和 GPTQ 量化有什么不同？",
         "keywords": ["激活值", "缩放", "Hessian", "INT4"]},
        {"question": "Python 中装饰器是什么？写一个简单例子。",
         "keywords": ["高阶函数", "增强", "注解", "元编程"]},
        {"question": "什么是 Python GIL？它有什么影响？",
         "keywords": ["全局锁", "线程", "CPython", "并行"]},
        {"question": "FastAPI 和 Flask 的主要区别是什么？",
         "keywords": ["异步", "自动文档", "性能", "校验"]},
        {"question": "Docker 中 COPY 和 ADD 指令有什么区别？",
         "keywords": ["复制", "URL", "解压", "构建缓存"]},
        {"question": "什么是模型幻觉？有哪些缓解方法？",
         "keywords": ["事实错误", "RAG", "微调", "约束"]},
        {"question": "LLM 推理时 temperature 参数的作用是什么？",
         "keywords": ["随机性", "采样", "多样性", "贪婪"]},
        {"question": "什么是 Prompt Engineering？有哪些常见技巧？",
         "keywords": ["提示", "Few-shot", "CoT", "指令"]},
        {"question": "LangChain 中 Chain 和 Agent 的核心区别是什么？",
         "keywords": ["固定流程", "动态", "工具", "决策"]},
        {"question": "什么是 PagedAttention？解决了什么问题？",
         "keywords": ["分页", "显存", "KV", "碎片"]},
        {"question": "Batch Normalization 和 Layer Normalization 的区别？",
         "keywords": ["维度", "Batch", "依赖", "特征"]},
        {"question": "什么是梯度消失和梯度爆炸？如何缓解？",
         "keywords": ["指数", "残差", "裁剪", "激活"]},
        {"question": "Adam 优化器和 SGD 相比有什么优势？",
         "keywords": ["自适应", "动量", "学习率", "收敛"]},
        {"question": "Fine-tuning 和 In-Context Learning 有什么区别？",
         "keywords": ["权重", "示例", "更新", "泛化"]},
        {"question": "什么是 Tool Calling？LLM 如何决定调用哪个工具？",
         "keywords": ["结构化", "JSON", "函数", "参数"]},
    ]

    with open("./data/test_50.json", "w", encoding="utf-8") as f:
        json.dump(test_set, f, ensure_ascii=False, indent=2)
    print(f"  测试集: {len(test_set)} 条，已保存到 ./data/test_50.json")

    # 打印一条示例数据
    print(f"\n示例数据:")
    print(dataset[0]["text"][:200])
    print("...")

if __name__ == "__main__":
    main()
