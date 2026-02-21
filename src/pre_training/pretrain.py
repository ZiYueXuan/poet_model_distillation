import torch
from datasets import load_from_disk
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling
)

MODEL_PATH = "/root/autodl-tmp/models/deepseek-r1-8b"
PACKED_DATA_PATH = "/root/autodl-tmp/poet_model_distillation/resource/packed_poems"
OUT_DIR = "/root/autodl-tmp/poet_model_distillation/resource/deepseek-r1-pretrain"

if __name__ == '__main__':
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        dtype=torch.bfloat16,
        trust_remote_code=True,
    )

    model.config.use_cache = False
    model.gradient_checkpointing_enable()

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

        # 批次配置
        per_device_train_batch_size=1,
        gradient_accumulation_steps=16,

        # 训练规模
        num_train_epochs=1,
        max_steps=-1,

        # 优化器
        learning_rate=1e-4,
        weight_decay=0.01,
        adam_beta1=0.9,
        adam_beta2=0.95,
        adam_epsilon=1e-8,

        # 学习率调度
        lr_scheduler_type="cosine",
        warmup_steps=1000,
        warmup_ratio=0.02,

        # 训练配置
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        fp16=False,
        bf16=True,

        # 性能优化
        dataloader_num_workers=4,
        dataloader_pin_memory=True,
        torch_compile=True,  # 需要torch>=2.0
        optim="adamw_torch",

        # 日志与保存
        logging_steps=10,
        save_steps=2000,
        save_total_limit=3,
        save_strategy="steps",
        eval_steps=500,

        # 报告
        report_to="tensorboard",
        logging_dir=f"{OUT_DIR}/logs",

        # DeepSpeed
        deepspeed="/path/to/ds_zero3.json",

        # 其他
        remove_unused_columns=False,
        ddp_find_unused_parameters=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=data_collator,
    )

    trainer.train()

    trainer.save_model(OUT_DIR)
