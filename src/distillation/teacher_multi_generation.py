import json
import random

import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_PATH = "../../resource/models/deepseek-v2-lite"
BATCH_SIZE = 8
GENRES = ["五言绝句", "七言绝句", "五言律诗", "七言律诗"]

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_PATH,
    trust_remote_code=True,
    padding_side="left",
)

# 确保 pad_token 存在
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    print(f"设置 pad_token 为 eos_token (id: {tokenizer.pad_token_id})")

# 设置生成时的 pad_token_id（避免每次生成时的警告）
pad_token_id = tokenizer.pad_token_id
eos_token_id = tokenizer.eos_token_id

model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True
)


def build_prompts_A(lines: list[str]):
    return [
        f"""
        你是古典诗词创作助手。

        任务：对出工整下联。
        要求：
        1. 只输出下联
        2. 不要解释
        3. 不要出现多余文字
        4. 字数与上联一致
        
        上联：{line}
        下联：
        """
        for line in lines
    ]


def build_prompts_B(first_sentences: list[str]):
    return [
        f"""
        你是古典诗词创作助手。

        任务：请根据提供的首句：“{first_sentence}”，生成一首{"五言" if len(first_sentence) == 5 else "七言"} {random.choice(["绝句", "律诗"])}。
        要求：
        1. 只输出诗的正文
        2. 不要解释
        3. 不要出现多余文字
        """
        for first_sentence in first_sentences
    ]


def build_prompts_C(key_words_list: list[list[str]]):
    return [
        f"""
        你是古典诗词创作助手。
        任务：根据提供的关键词：{'，'.join(key_words)}，生成一首{random.choice(GENRES)}。
        要求：
        1. 只输出诗的正文
        2. 不要解释
        3. 不要出现多余文字
        """
        for key_words in key_words_list
    ]


def loading_data_for_lines(file_path: str) -> list[str]:
    """
    加载prepare文件夹下txt文档里的句子等，存在列表里，主要用于prompts的构建
    """
    lines = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            lines.append(line.strip())
    return lines


def loading_data_for_keywords(file_path: str) -> list[list[str]]:
    """
    加载prepare文件夹下txt文档里的关键词，存在列表里，主要用于prompts的构建
    """
    key_words_list = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            key_words = line.strip().split("   ")
            key_words_list.append(key_words)
    return key_words_list


def build_prompt_batches(prompts, batch_size, max_new_tokens, temp):
    prompt_batches = []

    for i in range(0, len(prompts), batch_size):
        batch_prompts = prompts[i:i + batch_size]
        prompt_batches.append({
            "prompts": batch_prompts,
            "max_tokens": max_new_tokens,
            "temp": temp
        })

    return prompt_batches


def generate_batch(prompts, max_new_tokens, temp):
    inputs = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512  # 防止过长的输入
    ).to(model.device)

    # 生成时指定 pad_token_id 和 eos_token_id 以避免警告
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temp,
            top_p=0.9,
            pad_token_id=pad_token_id,  # 明确指定，避免警告
            eos_token_id=eos_token_id,  # 明确指定
            repetition_penalty=1.1,  # 添加重复惩罚，提高生成质量
        )

    # 解码，只返回生成的部分
    texts = tokenizer.batch_decode(out, skip_special_tokens=True)
    return [t[len(p):] for t, p in zip(texts, prompts)]


def get_prompts_response(batched_prompts, output_path: str):
    for prompts in tqdm(batched_prompts):
        outputs = generate_batch(prompts["prompts"], prompts["max_tokens"], prompts["temp"])
        for p, o in zip(prompts["prompts"], outputs):
            with open(output_path, "a", encoding="utf-8") as f:
                f.write(json.dumps({"prompt": p, "response": o}, ensure_ascii=False) + "\n")


if __name__ == '__main__':
    print("正在根据上联生成下联...")
    prompts_A = build_prompts_A(loading_data_for_lines("../../resource/prepare/shang_lian.txt"))
    prompt_A_batches = build_prompt_batches(prompts_A, BATCH_SIZE, 40, 0.9)
    get_prompts_response(prompt_A_batches, "../../resource/prepare/shang_lian.json")

    print("正在根据首句生成诗...")
    prompts_B = build_prompts_B(loading_data_for_lines("../../resource/prepare/first_sentence.txt"))
    prompt_B_batches = build_prompt_batches(prompts_B, BATCH_SIZE, 100, 0.9)
    get_prompts_response(prompt_B_batches, "../../resource/prepare/shou_ju.json")

    print("正在根据关键词生成诗...")
    prompts_C = build_prompts_C(loading_data_for_keywords("../../resource/prepare/keywords.txt"))
    prompt_C_batches = build_prompt_batches(prompts_C, BATCH_SIZE, 100, 0.9)
    get_prompts_response(prompt_C_batches, "../../resource/prepare/keywords.json")

    print("DONE!!!")
