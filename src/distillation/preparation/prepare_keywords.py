import csv
import random

from tqdm import tqdm
import jieba.posseg as pos


def extract_keywords(csv_file_path):
    """
    从csv文件中提取关键字
    """
    try:
        with open(csv_file_path, 'r', encoding='utf-8') as file:
            reader = csv.reader(file)

            # 读取所有行
            rows = list(reader)

            # 提取关键词：多字，最好是名词
            keywords_set = set()

            for i, row in tqdm(enumerate(rows), total=len(rows), desc="Processing rows"):
                if row[1] in ['唐', '宋', '明', '清']:
                    col6 = row[5]
                    words = pos.cut(col6)
                    for word, flag in words:
                        if flag in {'n', 'nr', 'ns', 'nt', 't'} and len(word) > 1:
                            keywords_set.add(word)

        return keywords_set

    except FileNotFoundError:
        print(f"错误：找不到文件 {csv_file_path}")
        print("请确认文件路径是否正确")
    except Exception as e:
        print(f"处理文件时出错: {e}")


# 使用示例
if __name__ == "__main__":
    keywords = extract_keywords('../../../resource/all.csv')
    keywords_list = random.sample(list(keywords), min(20000, len(keywords)))
    random.shuffle(keywords_list)
    output_file = '../../../resource/keywords.txt'
    with open(output_file, 'w', encoding='utf-8') as f:
        for keyword in keywords_list:
            f.write(keyword + '\n')
    print(f"完成！已生成 {len(keywords_list)} 个关键字，已写入 {output_file}中")
