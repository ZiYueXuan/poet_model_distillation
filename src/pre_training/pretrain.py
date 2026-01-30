import torch
from datasets import load_from_disk
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling
)

MODEL_PATH = "../../resource/models/deepseek-r1-7b"
PACKED_DATA_PATH = "../../resource/packed_poems"
OUT_DIR = "../../resource/deepseek-poetry-pretrain"


if __name__ == '__main__':

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True
    )

    dataset = load_from_disk(PACKED_DATA_PATH)

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=AutoTokenizer.from_pretrained(
            MODEL_PATH,
            trust_remote_code=True
        ),
        mlm=False
    )

    training_args = TrainingArguments(
        output_dir=OUT_DIR,

        # ===== batch相关 =====
        per_device_train_batch_size=2,
        gradient_accumulation_steps=32,  # 2×32×2GPU=128

        # ===== 训练规模 =====
        num_train_epochs=3,
        learning_rate=2e-5,
        weight_decay=0.1,
        warmup_ratio=0.03,

        # ===== 性能 =====
        bf16=True,
        logging_steps=20,
        save_steps=500,
        save_total_limit=2,

        # ===== deepspeed =====
        deepspeed="ds_zero3.json",

        # ===== 其它 =====
        lr_scheduler_type="cosine",
        report_to="none",
        dataloader_num_workers=4,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=data_collator,
    )

    trainer.train()

    trainer.save_model(OUT_DIR)
