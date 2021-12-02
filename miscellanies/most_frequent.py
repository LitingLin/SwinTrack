from collections import Counter


def get_most_frequent_item_from_list(items: list):
    occurence_count = Counter(items)
    return occurence_count.most_common(1)[0][0]


def get_most_frequent_items_from_list(items: list, n = 1):
    occurence_count = Counter(items)
    return occurence_count.most_common(n)
