import os.path
from pathlib import Path

import numpy as np
from datasets import Dataset
from tqdm import tqdm
from transformers import AutoTokenizer

DATA_DIR = '../../resource/cleaned_poems'
MODEL_PATH = '../../resource/models/deepseek-r1-8b'
BLOCK_SIZE = 512
CHUNK_CHARS = 50000
OUTPUT_DIR = '../../resource/packed_poems'



def load_cleaned_texts(cleaned_poem_dir):
    """
    读取所有txt文件内容
    """
    contents = []
    for path in Path(cleaned_poem_dir).glob('*.txt'):
        with open(path, 'r', encoding='utf-8') as f:
            row = f.read()
        contents.append(row)
    return contents


def safe_tokenize(text, tokenizer, chunk_chars=50000):
    """
    对单个超长字符串做分块tokenize
    """
    ids_all = []

    for i in range(0, len(text), chunk_chars):
        sub = text[i:i+chunk_chars]

        ids = tokenizer(
            sub,
            add_special_tokens=False
        )["input_ids"]

        ids_all.extend(ids)

    return ids_all

def build_packed_dataset(texts, tokenizer, block_size=512):
    """
    核心思想：
    1. 全部tokenize成一个长序列
    2. 连续切block_size
    3. 不padding，不浪费token
    """
    all_ids = []

    for text in tqdm(texts, desc="Tokenizing files"):
        ids = safe_tokenize(text, tokenizer, CHUNK_CHARS)
        all_ids.append(tokenizer.bos_token_id)
        all_ids.extend(ids)
        all_ids.append(tokenizer.eos_token_id)

    print(f'Total tokens: {len(all_ids)}')

    total_len = (len(all_ids) // block_size) * block_size
    all_ids = all_ids[:total_len]

    print(f'After packing tokens: {total_len:,}')
    print(f'Total sequences: {total_len // block_size:,}')

    arr = np.array(all_ids, dtype=np.int32)
    input_ids = arr.reshape((-1, block_size))
    dataset = Dataset.from_dict({
        'input_ids': input_ids.tolist(),
    })
    return dataset


def add_labels(examples):
    """
    给每个样本添加标签
    """
    examples['labels'] = examples['input_ids']
    return examples


if __name__ == '__main__':
    print('Loading tokenizer: ')
    try:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f'Model path {MODEL_PATH} does not exist')
        tokenizer = AutoTokenizer.from_pretrained(
            MODEL_PATH,
            trust_remote_code=True
        )
        # 检查tokenizer是否成功加载
        if tokenizer is None:
            raise ValueError('Tokenizer is None')
    except Exception as e:
        print(f'Error loading tokenizer: {e}')

    if tokenizer.eos_token is None:
        tokenizer.add_special_tokens({'eos_token': '</s>'})

    print('Loading texts...')
    texts = load_cleaned_texts(DATA_DIR)

    print('Packing dataset...')
    dataset = build_packed_dataset(texts, tokenizer, BLOCK_SIZE)
    dataset = dataset.map(add_labels, num_proc=4)

    print('Saving datasets...')
    dataset.save_to_disk(OUTPUT_DIR)
    print('Done!')
