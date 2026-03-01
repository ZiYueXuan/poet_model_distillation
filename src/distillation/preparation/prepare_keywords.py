import os
import random
import re

import jieba.posseg as pseg
from tqdm import tqdm

from src.utils.extract_sentence import split_poem_lines, is_lv_shi


def extract_keywords(file_path: str) -> list[list]:
    """
    从文件中提取关键字
    """
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()

        # 使用正则匹配每个条目块：从"题目："开始，直到下一个"题目："或文件结束
        pattern = r'(题目：.*?)(?=题目：|$)'
        poem_blocks = re.findall(pattern, content, re.DOTALL)
        keywords_list = []
        for block in tqdm(poem_blocks):
            body_match = re.search(r'正文：(.*)', block, re.DOTALL)
            if not body_match:                continue
            body_text = body_match.group(1).strip()
            poem_lines = split_poem_lines(body_text)
            selected_words = []
            if is_lv_shi(poem_lines):
                for sentence in poem_lines:
                    words = pseg.cut(sentence.strip())

                    valid_words = []
                    for word, flag in words:
                        # 保留名词、专名
                        if flag in {'n', 'nr', 'ns'} and len(word) >= 2:
                            valid_words.append(word)

                    # 随机选择最多5个关键词
                    if len(valid_words) > 5:
                        selected_words = random.sample(valid_words, 5)
                    else:
                        selected_words = valid_words
            if len(selected_words) > 0:
                keywords_list.append(selected_words)
        return keywords_list


    except FileNotFoundError:
        print(f"错误：找不到文件 {file_path}")
        print("请确认文件路径是否正确")
    except Exception as e:
        print(f"处理文件时出错: {e}")


# 使用示例
if __name__ == "__main__":
    base_dir = '../../../resource/cleaned_poems'  # 文件夹路径
    files = {
        'tang2.txt': 4000,
        'song4.txt': 4000,
        'ming2.txt': 6000,
        'qing1.txt': 6000
    }
    # 设置随机种子以保证可重复性
    random.seed(42)
    all_keywords = []
    for file_name, needed in files.items():
        filepath = os.path.join(base_dir, file_name)
        if not os.path.exists(filepath):
            print(f"警告：文件 {filepath} 不存在，跳过。")
            continue

        keywords = extract_keywords(filepath)
        print(f"{file_name}：共提取到 {len(keywords)} 个关键词")

        if len(keywords) < needed:
            print(f"警告：{file_name} 中关键词数不足 {needed}，将采用有放回抽样（可能包含重复）。")
            sample = random.choices(keywords, k=needed)
        else:
            sample = random.sample(keywords, needed)

        all_keywords.extend(sample)

    random.shuffle(all_keywords)
    output_file = '../../../resource/prepare/keywords.txt'
    print(f"准备写入文件 {output_file}")
    with open(output_file, 'w', encoding='utf-8') as f:
        for keyword_lists in all_keywords:
            for key in keyword_lists:
                f.write(key + '   ')
            f.write('\n')

    print(f"DONE!!!")
