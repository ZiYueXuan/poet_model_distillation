from pypinyin import pinyin, Style


def convert_chinese_to_pinyin(text):
    """将中文文本转换为带声调的拼音，用下划线连接"""
    if not text:
        return ''
    words = pinyin(text, style=Style.TONE3, heteronym=False)
    return '_'.join(word[0] for word in words)