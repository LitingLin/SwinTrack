# https://github.com/ianlini/flatten-dict/blob/master/src/flatten_dict/flatten_dict.py

from collections.abc import Mapping
from .reducers import tuple_reducer, path_reducer, dot_reducer, underscore_reducer
from .splitters import tuple_splitter, path_splitter, dot_splitter, underscore_splitter


REDUCER_DICT = {
    "tuple": tuple_reducer,
    "path": path_reducer,
    "dot": dot_reducer,
    "underscore": underscore_reducer,
}

SPLITTER_DICT = {
    "tuple": tuple_splitter,
    "path": path_splitter,
    "dot": dot_splitter,
    "underscore": underscore_splitter,
}


def flatten(
    d,
    reducer="tuple",
    inverse=False,
    max_flatten_depth=None,
    enumerate_types=(),
    keep_empty_types=(),
):
    """Flatten `Mapping` object.
    Parameters
    ----------
    d : dict-like object
        The dict that will be flattened.
    reducer : {'tuple', 'path', 'underscore', 'dot', Callable}
        The key joining method. If a `Callable` is given, the `Callable` will be
        used to reduce.
        'tuple': The resulting key will be tuple of the original keys.
        'path': Use `os.path.join` to join keys.
        'underscore': Use underscores to join keys.
        'dot': Use dots to join keys.
    inverse : bool
        Whether you want invert the resulting key and value.
    max_flatten_depth : Optional[int]
        Maximum depth to merge.
    enumerate_types : Sequence[type]
        Flatten these types using `enumerate`.
        For example, if we set `enumerate_types` to ``(list,)``,
        `list` indices become keys: ``{'a': ['b', 'c']}`` -> ``{('a', 0): 'b', ('a', 1): 'c'}``.
    keep_empty_types : Sequence[type]
        By default, ``flatten({1: 2, 3: {}})`` will give you ``{(1,): 2}``, that is, the key ``3``
        will disappear.
        This is also applied for the types in `enumerate_types`, that is,
        ``flatten({1: 2, 3: []}, enumerate_types=(list,))`` will give you ``{(1,): 2}``.
        If you want to keep those empty values, you can specify the types in `keep_empty_types`:
        >>> flatten({1: 2, 3: {}}, keep_empty_types=(dict,))
        {(1,): 2, (3,): {}}
    Returns
    -------
    flat_dict : dict
    """
    enumerate_types = tuple(enumerate_types)
    flattenable_types = (Mapping,) + enumerate_types
    if not isinstance(d, flattenable_types):
        raise ValueError(
            "argument type %s is not in the flattenalbe types %s"
            % (type(d), flattenable_types)
        )

    # check max_flatten_depth
    if max_flatten_depth is not None and max_flatten_depth < 1:
        raise ValueError("max_flatten_depth should not be less than 1.")

    if isinstance(reducer, str):
        reducer = REDUCER_DICT[reducer]
    flat_dict = {}

    def _flatten(_d, depth, parent=None):
        key_value_iterable = (
            enumerate(_d) if isinstance(_d, enumerate_types) else _d.items()
        )
        has_item = False
        for key, value in key_value_iterable:
            has_item = True
            flat_key = reducer(parent, key)
            if isinstance(value, flattenable_types) and (
                max_flatten_depth is None or depth < max_flatten_depth
            ):
                # recursively build the result
                has_child = _flatten(value, depth=depth + 1, parent=flat_key)
                if has_child or not isinstance(value, keep_empty_types):
                    # ignore the key in this level because it already has child key
                    # or its value is empty
                    continue

            # add an item to the result
            if inverse:
                flat_key, value = value, flat_key
            if flat_key in flat_dict:
                raise ValueError("duplicated key '{}'".format(flat_key))
            flat_dict[flat_key] = value

        return has_item

    _flatten(d, depth=1)
    return flat_dict


def nested_set_dict(d, keys, value):
    """Set a value to a sequence of nested keys.
    Parameters
    ----------
    d : Mapping
    keys : Sequence[str]
    value : Any
    """
    assert keys
    key = keys[0]
    if len(keys) == 1:
        if key in d:
            raise ValueError("duplicated key '{}'".format(key))
        d[key] = value
        return
    d = d.setdefault(key, {})
    nested_set_dict(d, keys[1:], value)


def unflatten(d, splitter="tuple", inverse=False):
    """Unflatten dict-like object.
    Parameters
    ----------
    d : dict-like object
        The dict that will be unflattened.
    splitter : {'tuple', 'path', 'underscore', 'dot', Callable}
        The key splitting method. If a Callable is given, the Callable will be
        used to split `d`.
        'tuple': Use each element in the tuple key as the key of the unflattened dict.
        'path': Use `pathlib.Path.parts` to split keys.
        'underscore': Use underscores to split keys.
        'dot': Use underscores to split keys.
    inverse : bool
        Whether you want to invert the key and value before flattening.
    Returns
    -------
    unflattened_dict : dict
    """
    if isinstance(splitter, str):
        splitter = SPLITTER_DICT[splitter]

    unflattened_dict = {}
    for flat_key, value in d.items():
        if inverse:
            flat_key, value = value, flat_key
        key_tuple = splitter(flat_key)
        nested_set_dict(unflattened_dict, key_tuple, value)

    return unflattened_dict
