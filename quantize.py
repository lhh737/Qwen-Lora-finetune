"""
模型量化脚本

将 LoRA 适配器合并到基座模型后，量化导出为 INT4 格式。
量化后模型体积减少约 60%，便于部署。

流程：
  1. 加载基座模型 + LoRA 适配器
  2. 合并权重 (merge and unload)
  3. 应用 GPTQ/AWQ 量化
  4. 导出量化模型

使用方法：
  python quantize.py                                  # 1.5B + GPTQ
  python quantize.py --model_id Qwen/Qwen2.5-7B       # 7B
  python quantize.py --method awq                     # 改用 AWQ

注意：
  - 1.5B 可以在 Colab T4 上完成量化
  - 7B 量化需要较多显存，建议在 24GB+ 的环境运行
"""

import os, torch, argparse
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

def main():
    parser = argparse.ArgumentParser(description="模型量化导出")
    parser.add_argument("--model_id", default="Qwen/Qwen2.5-1.5B",
                        help="基座模型 ID")
    parser.add_argument("--lora_path", default="./output/lora_final",
                        help="LoRA 适配器路径")
    parser.add_argument("--output_dir", default="./output/qwen_int4",
                        help="量化模型输出路径")
    parser.add_argument("--method", default="gptq", choices=["gptq", "awq"],
                        help="量化方法")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Step 1: 加载 tokenizer
    print("[1/4] 加载 tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)

    # Step 2: 加载基座模型和 LoRA，合并权重
    print("[2/4] 加载基座模型 + LoRA，合并权重...")
    base = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )
    lora = PeftModel.from_pretrained(base, args.lora_path)
    merged = lora.merge_and_unload()  # 合并 LoRA 权重到基座模型
    print("  合并完成 ✓")

    # Step 3: 量化
    print(f"[3/4] 应用 {args.method.upper()} 量化 (INT4)...")

    if args.method == "awq":
        # AWQ 量化
        try:
            from awq import AutoAWQForCausalLM
            model = AutoAWQForCausalLM.from_pretrained(args.model_id)
            model.quantize(tokenizer, quant_config={"bits": 4, "group_size": 128})
            model.save_quantized(args.output_dir)
            print("  AWQ 量化完成 ✓")
        except ImportError:
            print("  未安装 autoawq，请执行: pip install autoawq")
            return
    else:
        # GPTQ 量化（默认）
        try:
            from optimum.gptq import GPTQQuantizer
            quantizer = GPTQQuantizer(bits=4, dataset="c4",
                                      block_name_to_quantize="model.layers")
            quantized = quantizer.quantize_model(merged, tokenizer)
            quantized.save_pretrained(args.output_dir)
            print("  GPTQ 量化完成 ✓")
        except ImportError:
            print("  未安装 optimum/auto-gptq，请执行: pip install optimum auto-gptq")
            return

    tokenizer.save_pretrained(args.output_dir)

    # Step 4: 验证
    print("[4/4] 验证量化模型...")
    model_size = sum(os.path.getsize(os.path.join(dp, f))
                     for dp, _, fn in os.walk(args.output_dir)
                     for f in fn) / (1024**3)
    print(f"  模型目录大小: {model_size:.2f} GB")

    # 测试推理
    test_model = AutoModelForCausalLM.from_pretrained(
        args.output_dir, device_map="auto", trust_remote_code=True
    )
    inputs = tokenizer("你好", return_tensors="pt").to(test_model.device)
    out = test_model.generate(**inputs, max_new_tokens=20)
    print(f"  推理测试输出: {tokenizer.decode(out[0], skip_special_tokens=True)}")
    del test_model

    print(f"\n量化模型已保存到: {args.output_dir}")
    print(f"量化前 (FP16) 约 {model_size * 4:.2f} GB → 量化后约 {model_size:.2f} GB")

if __name__ == "__main__":
    main()
