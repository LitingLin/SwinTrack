from datasets.filter._common import _BaseFilter


def filter_list_serialize(filters: list):
    return [filter_.serialize() for filter_ in filters]


def filter_list_deserialize(filters_serialized: list):
    return [_BaseFilter.deserialize(filter_serialized) for filter_serialized in filters_serialized]
