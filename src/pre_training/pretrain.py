import torch
from datasets import load_from_disk
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    TrainerCallback
)
from peft import LoraConfig, get_peft_model

MODEL_PATH = "../../resource/models/deepseek-r1-8b"
DATA_PATH = "../../resource/packed_poems"
OUTPUT_DIR = "../../resource/output/poetry_dapt"


class LossOnlyCallback(TrainerCallback):
    def __init__(self, log_path="loss.log"):
        self.log_path = log_path

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None:
            return

        if "loss" in logs:
            with open(self.log_path, "a") as f:
                f.write(f"{state.global_step},   {logs['loss']}\n")


def main():
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_PATH,
        trust_remote_code=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True
    )

    # LoRA
    lora_config = LoraConfig(
        r=64,
        lora_alpha=128,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj"
        ],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    print("Loading packed dataset...")
    dataset = load_from_disk(DATA_PATH)

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        num_train_epochs=2,
        learning_rate=1e-5,
        warmup_ratio=0.03,
        logging_steps=10,
        save_steps=500,
        save_total_limit=2,
        bf16=True,
        deepspeed="ds_config.json",
        remove_unused_columns=False,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        callbacks=[LossOnlyCallback()]
    )

    # 尝试从最新 checkpoint 恢复
    # === 新增：自动查找最新 checkpoint 并恢复 ===
    import os
    last_checkpoint = None
    if os.path.exists(OUTPUT_DIR):
        # 查找所有以 "checkpoint-" 开头的目录（DeepSpeed 默认命名）
        checkpoints = [
            d for d in os.listdir(OUTPUT_DIR)
            if os.path.isdir(os.path.join(OUTPUT_DIR, d)) and d.startswith("checkpoint-")
        ]
        if checkpoints:
            # 按数字排序取最新（如 checkpoint-3500 > checkpoint-3000）
            checkpoints.sort(key=lambda x: int(x.split("-")[1]))
            last_checkpoint = os.path.join(OUTPUT_DIR, checkpoints[-1])
            print(f"✅ Found latest checkpoint: {last_checkpoint}")
        else:
            print("⚠️  No checkpoint found, starting from scratch.")
    else:
        print("⚠️  Output directory does not exist, starting from scratch.")

    # 启动训练（自动恢复或从头开始）
    trainer.train(resume_from_checkpoint=last_checkpoint)
    trainer.save_model(OUTPUT_DIR)


if __name__ == "__main__":
    main()
