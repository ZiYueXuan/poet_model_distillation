import random


def get_random_genres(num:int):
    """
    随即从五言绝句、七言绝句、五言律诗、七言律诗里选一个体裁，加入到体裁列表中
    :return:
    """
    genres = []
    for i in range(num):
        genres.append(random.choice(["五言绝句", "七言绝句", "五言律诗", "七言律诗"]))
    return genres