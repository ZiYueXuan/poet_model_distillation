import logging
import os
import re

import pandas as pd

from tqdm import tqdm

from src.utils.convert_chinese_to_pinyin import convert_chinese_to_pinyin

logger = logging.getLogger(__name__)


def clean(text):
    """
    文本格式化
    """
    return re.sub(r"\(.*?\)", "", text).replace(" ", "").strip()


def data_cleaning():
    """
    读取all.csv文件，将诗歌按照朝代整理成
    题目：{title}
    朝代：{dynasty}
    作者：{author}
    正文：{content}
    每个朝代写一个文件
    """
    df = pd.read_csv('../../resource/all.csv', header=None, dtype=str)

    # 朝代数字替换
    df[1] = df[1].astype(str).str.strip().apply(lambda x: 'min2_guo2' if x.isdigit() else x)  # 朝代
    df[2] = df[2].astype(str).str.strip()  # 作者
    df[0] = df[0].astype(str).str.strip()  # 标题
    df[3] = df[3].astype(str).str.strip()  # 正文

    for (dynasty, author), group in tqdm(df.groupby([1, 2])):
        dynasty_en = convert_chinese_to_pinyin(dynasty)
        dir_name = f"../../resource/cleaned_poems"
        file_name = f"{dynasty_en}.txt"
        with open(os.path.abspath(os.path.join(dir_name, file_name)), "a", encoding="utf-8") as f:
            try:
                for _, row in group.iterrows():
                    lines = [
                        f"题目：{clean(row[0])}",
                        f"朝代：{dynasty}",
                        f"作者：{clean(row[2])}",
                        f"正文：{clean(row[3])}",
                        ""
                    ]
                    f.write('\n'.join(lines))
            except Exception as e:
                logger.error(f"Error processing {dynasty} {author}: {e}")
                continue


if __name__ == '__main__':
    data_cleaning()
