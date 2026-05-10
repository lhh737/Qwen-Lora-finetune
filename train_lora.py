"""
LoRA 微调训练脚本

基于 HuggingFace TRL 的 SFTTrainer，对 Qwen2.5 进行 LoRA 指令微调。

使用方法：
  python train_lora.py                     # 默认参数 (1.5B)
  python train_lora.py --model_id Qwen/Qwen2.5-7B   # 7B 模型
  python train_lora.py --model_id Qwen/Qwen2.5-7B --use_qlora  # 7B + QLoRA

输出：
  ./output/lora_final/     LoRA 适配器权重 + tokenizer
"""

import os, sys, time, torch, argparse
from datasets import load_from_disk
from transformers import (
    AutoModelForCausalLM, AutoTokenizer, TrainingArguments, BitsAndBytesConfig
)
from peft import LoraConfig
from trl import SFTTrainer

# ==================== 配置 ====================
def parse_args():
    parser = argparse.ArgumentParser(description="Qwen2.5 LoRA 指令微调")
    parser.add_argument("--model_id", type=str, default="Qwen/Qwen2.5-1.5B",
                        help="HuggingFace 模型 ID")
    parser.add_argument("--data_path", type=str, default="./data/alpaca_5k",
                        help="训练数据路径")
    parser.add_argument("--output_dir", type=str, default="./output/lora_final",
                        help="模型保存路径")
    parser.add_argument("--use_qlora", action="store_true",
                        help="使用 QLoRA (4-bit 加载)")
    parser.add_argument("--batch_size", type=int, default=4,
                        help="per_device_train_batch_size")
    parser.add_argument("--grad_accum", type=int, default=4,
                        help="gradient_accumulation_steps")
    parser.add_argument("--epochs", type=int, default=2,
                        help="训练轮数")
    parser.add_argument("--lr", type=float, default=2e-4,
                        help="学习率")
    parser.add_argument("--lora_rank", type=int, default=16,
                        help="LoRA rank")
    parser.add_argument("--max_seq_length", type=int, default=1024,
                        help="最大序列长度")
    return parser.parse_args()

def main():
    args = parse_args()
    os.makedirs("./output", exist_ok=True)

    is_qlora = args.use_qlora or "7B" in args.model_id

    print(f"配置:")
    print(f"  模型: {args.model_id}")
    print(f"  QLoRA: {is_qlora}")
    print(f"  LoRA rank: {args.lora_rank}")
    print(f"  训练数据: {args.data_path}")
    print(f"  Epochs: {args.epochs}")
    print(f"  学习率: {args.lr}")
    print(f"  Batch (等效): {args.batch_size * args.grad_accum}\n")

    # ── 1. 加载 tokenizer ──
    print("[1/5] 加载 tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # ── 2. 加载训练数据 ──
    print("[2/5] 加载训练数据...")
    dataset = load_from_disk(args.data_path)
    print(f"  训练样本数: {len(dataset)}")

    # ── 3. 加载模型 ──
    print("[3/5] 加载模型...")
    if is_qlora:
        print("  使用 QLoRA: 4-bit 加载基座模型")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            args.model_id,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            args.model_id,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
        )

    model.config.use_cache = False  # 训练时不需要 KV Cache

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  模型参数量: {total_params/1e9:.2f}B")
    print(f"  可训练参数量: {trainable_params/1e6:.2f}M")

    # ── 4. 配置 LoRA ──
    print("[4/5] 配置 LoRA...")
    lora_config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_rank * 2,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )

    # ── 5. 训练 ──
    print("[5/5] 开始训练...\n")
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        logging_steps=10,
        save_strategy="epoch",
        save_total_limit=2,
        report_to="none",
        fp16=True,
        dataloader_num_workers=2,
        remove_unused_columns=False,
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=args.max_seq_length,
        packing=True,
    )

    t0 = time.time()
    trainer.train()
    elapsed = time.time() - t0

    # 保存模型
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    # 打印训练摘要
    final_loss = trainer.state.log_history[-1].get("train_loss", 0)
    print(f"\n{'='*50}")
    print(f"  训练完成")
    print(f"  模型: {args.model_id}")
    print(f"  耗时: {elapsed/60:.1f} 分钟")
    print(f"  最终 loss: {final_loss:.4f}")
    print(f"  保存路径: {args.output_dir}")
    print(f"  参数总量: {sum(p.numel() for p in model.parameters())/1e6:.1f}M")
    print(f"{'='*50}")

if __name__ == "__main__":
    main()
