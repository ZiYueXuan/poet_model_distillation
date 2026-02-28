import os
import re
import random

from src.utils.extract_sentence import split_poem_lines, is_lv_shi


def extract_first_sentences_from_file(filepath):
    """
    从单个文件中提取所有律诗的首句
    文件格式：每个诗条由"题目："、"朝代："、"作者："、"正文："组成，条目连续无空行分隔
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()

    # 使用正则匹配每个条目块：从"题目："开始，直到下一个"题目："或文件结束
    pattern = r'(题目：.*?)(?=题目：|$)'
    poem_blocks = re.findall(pattern, content, re.DOTALL)

    first_sentences = []
    for block in poem_blocks:
        body_match = re.search(r'正文：(.*)', block, re.DOTALL)
        if not body_match:
            continue
        body_text = body_match.group(1).strip()
        poem_lines = split_poem_lines(body_text)
        if is_lv_shi(poem_lines):
            first_sentences.append(poem_lines[0])   # 取首句

    return first_sentences

if __name__ == '__main__':
    base_dir = '../../../resource/cleaned_poems'          # 文件夹路径
    files = {
        'tang2.txt': 7500,
        'song4.txt': 7500,
        'ming2.txt': 10000,
        'qing1.txt': 10000
    }
    all_first_sentences = []

    # 设置随机种子以保证可重复性
    random.seed(42)

    for file_name, needed in files.items():
        filepath = os.path.join(base_dir, file_name)
        if not os.path.exists(filepath):
            print(f"警告：文件 {filepath} 不存在，跳过。")
            continue

        first_sentences = extract_first_sentences_from_file(filepath)
        print(f"{file_name}：共提取到 {len(first_sentences)} 个律诗首句")

        if len(first_sentences) < needed:
            print(f"警告：{file_name} 中首句数不足 {needed}，将采用有放回抽样（可能包含重复）。")
            sample = random.choices(first_sentences, k=needed)
        else:
            sample = random.sample(first_sentences, needed)

        all_first_sentences.extend(sample)

    random.shuffle(all_first_sentences)       # 打乱最终顺序

    output_file = '../../../resource/first_sentence.txt'
    with open(output_file, 'w', encoding='utf-8') as f:
        for line in all_first_sentences:
            f.write(line + '\n')

    print(f"完成！共生成 {len(all_first_sentences)} 条律诗首句，已写入 {output_file}")