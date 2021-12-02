import collections.abc


def collate_batch_list(batch_list):
    elem = batch_list[0]
    if isinstance(elem, collections.abc.Sequence):
        return tuple(map(tuple, zip(*batch_list)))
    elif isinstance(elem, collections.abc.Mapping):
        return {key: tuple(d[key] for d in batch_list) for key in elem.keys()}
    else:
        raise NotImplementedError(f'Unsupported data type {elem}')
