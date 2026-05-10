"""
AWQ 量化：合并 LoRA 权重 → 4-bit 量化 → 导出
减小模型体积，方便部署
"""
import os, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

MODEL_NAME = "Qwen/Qwen2.5-1.5B"       # 也可以是 7B（需要更多显存）
LORA_PATH = "./output/lora_final"
OUTPUT_DIR = "./output/qwen_int4"

# Step 1: 加载基座 + LoRA → 合并
print("[1/3] 加载基座模型...")
base = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME, torch_dtype=torch.float16, device_map="auto", trust_remote_code=True
)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)

print("[2/3] 加载 LoRA 权重并合并...")
lora = PeftModel.from_pretrained(base, LORA_PATH)
merged = lora.merge_and_unload()
print("  合并完成 ✓")

# Step 2: 量化
print("[3/3] 应用 AWQ 量化...")
try:
    from awq import AutoAWQForCausalLM
    model = AutoAWQForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float16)
    model.quantize(tokenizer, quant_config={"bits": 4, "group_size": 128})
    model.save_quantized(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"  量化完成 ✓ 已保存到 {OUTPUT_DIR}")
    print(f"  模型体积约: {sum(os.path.getsize(os.path.join(dp, f)) for dp, _, fn in os.walk(OUTPUT_DIR) for f in fn) / 1e9:.2f} GB")
except ImportError:
    print("  未安装 autoawq，使用 GPTQ 替代...")
    !pip install -q optimum auto-gptq
    from optimum.gptq import GPTQQuantizer
    quantizer = GPTQQuantizer(bits=4, dataset="c4")
    quantized = quantizer.quantize_model(merged, tokenizer)
    quantized.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"  GPTQ 量化完成 ✓ 已保存到 {OUTPUT_DIR}")

print("\n你可以用以下代码测试量化后的模型：")
print("""
from transformers import AutoModelForCausalLM, AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("./output/qwen_int4")
model = AutoModelForCausalLM.from_pretrained("./output/qwen_int4", device_map="auto")
inputs = tokenizer("你好", return_tensors="pt").to(model.device)
out = model.generate(**inputs, max_new_tokens=50)
print(tokenizer.decode(out[0]))
""")
