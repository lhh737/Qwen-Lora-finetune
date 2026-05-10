"""
LoRA 微调：基于 Qwen2.5-7B 做指令微调（单次跑，不搞多组对比）
你入职后mentor给你的任务大概就是这样：把模型微调一下，跑通，看看效果
"""
import os, json, time
from datasets import load_from_disk
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from peft import LoraConfig
from trl import SFTTrainer

# ========== 配置 ==========
MODEL_NAME = "Qwen/Qwen2.5-7B"
DATA_PATH = "./data/alpaca_5k"
OUTPUT_DIR = "./output/lora_final"
BATCH_SIZE = 4
GRAD_ACCUM = 4       # 等效 batch = 16
EPOCHS = 2
LR = 2e-4
MAX_LEN = 1024

os.makedirs("./output", exist_ok=True)

# ========== 1. 加载 tokenizer ==========
print("[1/5] 加载 tokenizer ...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

# ========== 2. 加载数据 ==========
print("[2/5] 加载训练数据 ...")
dataset = load_from_disk(DATA_PATH)
print(f"  → {len(dataset)} 条训练样本")

# ========== 3. 加载模型 ==========
print("[3/5] 加载基座模型 ...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype="auto",    # 自动选 FP16/BF16
    device_map="auto",
    trust_remote_code=True,
)

# ========== 4. LoRA 配置 ==========
print("[4/5] 配置 LoRA ...")
lora_config = LoraConfig(
    r=16,                  # rank=16 是经过社区验证的稳妥选择
    lora_alpha=32,         # alpha = 2*r，常见实践
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

# ========== 5. 训练 ==========
print("[5/5] 开始训练 ...\n")
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRAD_ACCUM,
    num_train_epochs=EPOCHS,
    learning_rate=LR,
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
    max_seq_length=MAX_LEN,
    packing=True,
)

t0 = time.time()
trainer.train()
elapsed = time.time() - t0

# 保存
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

# 记录最终 loss
final_loss = trainer.state.log_history[-1].get("train_loss", 0)
print(f"\n{'='*50}")
print(f"  训练完成！")
print(f"  耗时: {elapsed/60:.1f} 分钟")
print(f"  最终 loss: {final_loss:.4f}")
print(f"  模型已保存到: {OUTPUT_DIR}")
print(f"{'='*50}")
