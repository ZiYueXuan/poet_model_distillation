import os
import re
import random

from src.utils.extract_sentence import split_poem_lines, is_lv_shi


def extract_sentences_from_file(filepath):
    """
    从单个文件中提取所有律诗的诗句
    文件格式：每个诗条由"题目："、"朝代："、"作者："、"正文："组成，条目连续无空行分隔
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()

    # 使用正则匹配每个条目块：从"题目："开始，直到下一个"题目："或文件结束
    # re.DOTALL 使点号匹配换行符，确保跨行匹配
    pattern = r'(题目：.*?)(?=题目：|$)'
    poem_blocks = re.findall(pattern, content, re.DOTALL)

    poem_sentences = []
    for block in poem_blocks:
        # 提取正文内容（位于"正文："之后，直到块结束）
        body_match = re.search(r'正文：(.*)', block, re.DOTALL)
        if not body_match:
            continue
        body_text = body_match.group(1).strip()
        # 将正文按标点拆分为句子
        poem_lines = split_poem_lines(body_text)
        if is_lv_shi(poem_lines):
            poem_sentences.extend(poem_lines)

    return poem_sentences

if __name__ == '__main__':
    base_dir = '../../../resource/cleaned_poems'          # 文件夹路径
    files = {
        'tang2.txt': 7500,
        'song4.txt': 7500,
        'ming2.txt': 10000,
        'qing1.txt': 10000
    }
    all_sentences = []

    # 设置随机种子以保证可重复性
    random.seed(42)

    for file_name, needed in files.items():
        filepath = os.path.join(base_dir, file_name)
        if not os.path.exists(filepath):
            print(f"警告：文件 {filepath} 不存在，跳过。")
            continue

        sentences = extract_sentences_from_file(filepath)
        print(f"{file_name}：共提取到 {len(sentences)} 个律诗句子")

        if len(sentences) < needed:
            print(f"警告：{file_name} 中句子数不足 {needed}，将采用有放回抽样（可能包含重复）。")
            sample = random.choices(sentences, k=needed)
        else:
            sample = random.sample(sentences, needed)

        all_sentences.extend(sample)

    random.shuffle(all_sentences)       # 打乱最终顺序

    output_file = '../../../resource/prepare/shang_lian.txt'
    with open(output_file, 'w', encoding='utf-8') as f:
        for line in all_sentences:
            f.write(line + '\n')

    print(f"完成！共生成 {len(all_sentences)} 条上联，已写入 {output_file}")
