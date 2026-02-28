import re


def count_hanzi(text):
    """统计字符串中的汉字个数（基于Unicode范围）"""
    return len(re.findall(r'[\u4e00-\u9fff]', text))

def split_poem_lines(text):
    """
    将正文按常见中文标点分割成独立的诗句
    标点包括：， 、 。 ？ ！ ； 等
    """
    lines = re.split(r'[，、。？！；]', text)
    lines = [line.strip() for line in lines if line.strip()]
    return lines

def is_lv_shi(sentences):
    """判断一组诗句是否为律诗：8句，每句5或7汉字"""
    if len(sentences) != 8:
        return False
    for s in sentences:
        if count_hanzi(s) not in (5, 7):
            return False
    return True