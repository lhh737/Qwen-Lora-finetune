"""
效果评估：微调前 vs 微调后，对比回答质量
在 30 条测试集上分别用基座模型和 LoRA 微调模型跑一遍
"""
import os, json, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

MODEL_NAME = "Qwen/Qwen2.5-7B"
LORA_PATH = "./output/lora_final"
TEST_PATH = "./data/test_100.json"
SAMPLE_SIZE = 30      # 测 30 条，够了

def load_test_data():
    with open(TEST_PATH, encoding="utf-8") as f:
        data = json.load(f)
    return data[:SAMPLE_SIZE]

def ask(model, tokenizer, question):
    """问一个问题，返回回答"""
    prompt = f"<|im_start|>user\n{question}\n<|im_end|>\n<|im_start|>assistant\n"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=128,
            temperature=0.1,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )
    return tokenizer.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)

def simple_score(answer, reference):
    """简易评分：回答中包含参考关键字的比例"""
    if not answer or not reference:
        return 0
    ref_words = set(reference.split())
    ans_words = set(answer.split())
    if not ref_words:
        return 0
    overlap = ref_words & ans_words
    return len(overlap) / len(ref_words)

def main():
    test_data = load_test_data()
    print(f"测试集: {len(test_data)} 条\n")

    # ── 加载基座模型（微调前）──
    print("[1/3] 加载基座模型（微调前）...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype="auto", device_map="auto", trust_remote_code=True
    )
    base_model.eval()

    # ── 加载微调后模型 ──
    print("[2/3] 加载微调后模型（LoRA）...")
    lora_model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype="auto", device_map="auto", trust_remote_code=True
    )
    lora_model = PeftModel.from_pretrained(lora_model, LORA_PATH)
    lora_model.eval()

    # ── 逐条对比 ──
    print("[3/3] 逐条对比 ...\n")
    base_scores = []
    lora_scores = []
    better = 0
    worse = 0

    for i, item in enumerate(test_data):
        q = item["question"]
        ref = item["reference"]

        base_ans = ask(base_model, tokenizer, q)
        lora_ans = ask(lora_model, tokenizer, q)

        base_s = simple_score(base_ans, ref)
        lora_s = simple_score(lora_ans, ref)
        base_scores.append(base_s)
        lora_scores.append(lora_s)

        if lora_s > base_s:
            better += 1
        elif lora_s < base_s:
            worse += 1

        # 每 10 条打印一个示例
        if i % 10 == 0:
            print(f"── 问题 {i+1}: {q[:40]}...")
            print(f"  微调前: {base_ans[:80]}...")
            print(f"  微调后: {lora_ans[:80]}...")
            print(f"  得分: {base_s:.2f} → {lora_s:.2f}\n")

    # ── 汇总 ──
    avg_base = sum(base_scores) / len(base_scores)
    avg_lora = sum(lora_scores) / len(lora_scores)
    improve = (avg_lora - avg_base) / avg_base * 100 if avg_base > 0 else 0

    print(f"\n{'='*50}")
    print(f"  评估汇总 ({SAMPLE_SIZE} 条)")
    print(f"{'='*50}")
    print(f"  微调前平均得分: {avg_base:.3f}")
    print(f"  微调后平均得分: {avg_lora:.3f}")
    print(f"  提升: {improve:+.1f}%")
    print(f"  微调后更好: {better}/{SAMPLE_SIZE}")
    print(f"  微调后更差: {worse}/{SAMPLE_SIZE}")
    print(f"{'='*50}")

if __name__ == "__main__":
    main()
