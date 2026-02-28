import json, random, torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm

MODEL_PATH = "../../resource/models/deepseek-r1-14b"
OUT_FILE = "distill_80k.jsonl"
BATCH_SIZE = 8

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_PATH,
    trust_remote_code=True
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True
)


def build_prompts_A(lines: list[str]):
    return [
        f"请对出工整下联：\n上联：{line}\n下联："
        for line in lines
    ]


def build_prompts_B(first_sentences: list[str], genres: list[str]):
    return [
        f"请根据提供的首句：{first_sentence}，生成一首{genre}诗。"
        for first_sentence, genre in zip(first_sentences, genres)
    ]


def build_prompts_C(key_words_list: list[list[str]], genres: list[str]):
    return [
        f"请根据提供的关键词：{','.join(key_words)}，生成一首{genre}诗。"
        for (key_words, genre) in zip(key_words_list)
    ]


def generate_batch(prompts, max_new_tokens, temp):
    inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(model.device)
    out = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=temp,
        top_p=0.9
    )
    texts = tokenizer.batch_decode(out, skip_special_tokens=True)
    return [t[len(p):] for t,p in zip(texts,prompts)]


if __name__ == '__main__':
    with open(OUT_FILE, "w", encoding="utf-8") as f:
        for batch in tqdm(prompt_batches):
            outputs = generate_batch(batch["prompts"], batch["max_tokens"], batch["temp"])
            for p, o in zip(batch["prompts"], outputs):
                f.write(json.dumps({"prompt": p, "response": o}, ensure_ascii=False) + "\n")