# https://github.com/ianlini/flatten-dict/blob/master/src/flatten_dict/splitters.py

def tuple_splitter(flat_key):
    return flat_key


def path_splitter(flat_key):
    try:
        from pathlib import PurePath
    except ImportError:
        from pathlib2 import PurePath
    keys = PurePath(flat_key).parts
    return keys


def dot_splitter(flat_key):
    keys = tuple(flat_key.split("."))
    return keys


def underscore_splitter(flat_key):
    keys = tuple(flat_key.split("_"))
    return keys


def make_splitter(delimiter):
    """Create a reducer with a custom delimiter.
    Parameters
    ----------
    delimiter : str
        Delimiter to use to split keys.
    Returns
    -------
    f : Callable
        Callable that can be passed to ``unflatten``'s ``splitter`` argument.
    """

    def f(flat_key):
        keys = tuple(flat_key.split(delimiter))
        return keys

    return f