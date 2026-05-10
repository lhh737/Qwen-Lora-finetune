"""
效果评估脚本

对比基座模型和 LoRA 微调后模型在 20 道技术问答上的表现。
使用关键词命中率作为评估指标。

使用方法：
  python evaluate.py                                   # 默认 (1.5B)
  python evaluate.py --model_id Qwen/Qwen2.5-7B        # 7B 模型
  python evaluate.py --model_id Qwen/Qwen2.5-1.5B --lora_path ./output/lora_final

输出：
  控制台打印对比结果
"""

import json, torch, argparse
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

def ask(model, tokenizer, question, max_tokens=96):
    """向模型提问，返回回答"""
    prompt = f"<|im_start|>user\n{question}\n<|im_end|>\n<|im_start|>assistant\n"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=0.1,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )
    return tokenizer.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()

def hit_rate(answer, keywords):
    """计算关键词命中率：answer 中包含 keywords 的比例"""
    if not keywords:
        return 0.0
    ans_lower = answer.lower()
    hits = sum(1 for kw in keywords if kw.lower() in ans_lower)
    return hits / len(keywords)

def main():
    parser = argparse.ArgumentParser(description="评估微调效果")
    parser.add_argument("--model_id", default="Qwen/Qwen2.5-1.5B",
                        help="基座模型 ID")
    parser.add_argument("--lora_path", default="./output/lora_final",
                        help="LoRA 适配器路径")
    parser.add_argument("--test_path", default="./data/test_50.json",
                        help="测试集路径")
    args = parser.parse_args()

    # 加载测试集
    with open(args.test_path, encoding="utf-8") as f:
        test_data = json.load(f)
    questions = [item["question"] for item in test_data]
    keywords_list = [item.get("keywords", []) for item in test_data]

    print(f"测试集: {len(test_data)} 道题\n")

    # 加载基座模型
    print("[1/3] 加载基座模型（微调前）...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    base = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    ).eval()

    # 加载 LoRA 微调模型
    print("[2/3] 加载微调后模型...")
    lora = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )
    lora = PeftModel.from_pretrained(lora, args.lora_path).eval()

    # 逐题评估
    print("[3/3] 评估中...\n")
    results = []
    for i, q in enumerate(questions):
        base_ans = ask(base, tokenizer, q)
        lora_ans = ask(lora, tokenizer, q)
        kws = keywords_list[i]

        base_score = hit_rate(base_ans, kws)
        lora_score = hit_rate(lora_ans, kws)

        results.append((q, base_ans, lora_ans, base_score, lora_score))

        # 每 5 题打印一次详情
        if i % 5 == 0:
            direction = "↑" if lora_score > base_score else ("↓" if lora_score < base_score else "=")
            print(f"Q{i+1}: {q[:50]}...")
            print(f"  base: {base_ans[:60]}... [{base_score:.0%}]")
            print(f"  lora: {lora_ans[:60]}... [{lora_score:.0%}] {direction}\n")

    # 汇总
    base_avg = sum(r[3] for r in results) / len(results)
    lora_avg = sum(r[4] for r in results) / len(results)
    improve = (lora_avg - base_avg) / base_avg * 100 if base_avg > 0 else 0

    better = sum(1 for r in results if r[4] > r[3])
    worse = sum(1 for r in results if r[4] < r[3])

    print(f"\n{'='*55}")
    print(f"  评估汇总 ({len(results)} 题)")
    print(f"{'='*55}")
    print(f"  基座模型平均命中率:  {base_avg:.1%}")
    print(f"  微调后平均命中率:  {lora_avg:.1%}")
    print(f"  相对提升:          {improve:+.0f}%")
    print(f"  微调后更好的题数:  {better}/{len(results)}")
    print(f"  微调后更差的题数:  {worse}/{len(results)}")

    if lora_avg > base_avg:
        print(f"\n结论: LoRA 微调在该测试集上有效提升了回答质量 ✓")
    else:
        print(f"\n结论: 本次微调效果不显著。建议检查数据质量或增加训练量。")

if __name__ == "__main__":
    main()
