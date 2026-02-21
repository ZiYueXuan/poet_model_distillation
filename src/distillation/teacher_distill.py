import os

import torch
from datasets import load_from_disk
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

# ======================
# 配置区
# ======================
TEACHER_PATH = "/root/autodl-tmp/models/deepseek-r1-32b"
DATASET_PATH = "/root/autodl-tmp/poet_model_distillation/resource/packed_poems"
SAVE_DIR = "/root/autodl-tmp/distill_data"

BATCH_SIZE = 2
TOP_K = 8
MAX_LEN = 2048
SAVE_EVERY = 500   # 每多少 step 保存一次 chunk

os.makedirs(SAVE_DIR, exist_ok=True)

# ======================
# 加载 teacher
# ======================
tokenizer = AutoTokenizer.from_pretrained(TEACHER_PATH, trust_remote_code=True)

model = AutoModelForCausalLM.from_pretrained(
    TEACHER_PATH,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True,
)

model.eval()

# ======================
# 数据
# ======================
dataset = load_from_disk(DATASET_PATH)

# ======================
# 缓存容器
# ======================
buffer = []
chunk_id = 0

# ======================
# 推理
# ======================
for step in tqdm(range(0, len(dataset), BATCH_SIZE)):
    batch = dataset[step: step + BATCH_SIZE]["text"]

    inputs = tokenizer(
        batch,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=MAX_LEN
    ).to(model.device)

    with torch.no_grad():
        outputs = model(
            **inputs,
            output_hidden_states=True,
            use_cache=False
        )

    logits = outputs.logits              # [B, T, V]
    hidden = outputs.hidden_states[-1]   # [B, T, H]

    # ==========
    # top-k logits
    # ==========
    top_k_logits, top_k_ids = torch.topk(logits, TOP_K, dim=-1)

    # ==========
    # teacher greedy 文本
    # ==========
    pred_ids = torch.argmax(logits, dim=-1)
    pred_text = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)

    # ==========
    # 写入 buffer
    # ==========
    for i in range(len(batch)):
        item = {
            "text": batch[i],
            "teacher_text": pred_text[i],
            "topk_ids": top_k_ids[i].cpu().to(torch.int32).numpy(),
            "topk_logits": top_k_logits[i].cpu().to(torch.float16).numpy(),
            "hidden": hidden[i].cpu().to(torch.float16).numpy(),
        }
        buffer.append(item)

    # ==========
    # chunk 保存
    # ==========
    if len(buffer) >= SAVE_EVERY:
        torch.save(buffer, f"{SAVE_DIR}/chunk_{chunk_id}.pt")
        buffer = []
        chunk_id += 1

# 最后残余
if buffer:
    torch.save(buffer, f"{SAVE_DIR}/chunk_{chunk_id}.pt")

print("Teacher dumping finished")