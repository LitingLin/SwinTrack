import os
import hashlib
from miscellanies.slugify import slugify

_cache_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'cache'))


def get_dataset_attributes_cache_path():
    global _cache_dir
    return _cache_dir


def set_dataset_attributes_cache_path(path: str):
    global _cache_dir
    _cache_dir = path


def prepare_dataset_cache_path(dataset_class_name, dataset_name: str, dataset_filters: list, dataset_split):
    m = hashlib.md5()
    m.update(bytes(dataset_name, encoding='utf-8'))
    if dataset_filters is not None:
        m.update(bytes(str(dataset_filters), encoding='utf-8'))
    cache_path = get_dataset_attributes_cache_path()
    cache_path = os.path.join(cache_path, dataset_class_name)
    if dataset_filters is not None:
        cache_path = os.path.join(cache_path, 'filtered')
    os.makedirs(cache_path, exist_ok=True)
    cache_file_name_prefix = '{}-{}-{}'.format(slugify(dataset_name), str(dataset_split), m.digest().hex())
    return cache_path, cache_file_name_prefix
