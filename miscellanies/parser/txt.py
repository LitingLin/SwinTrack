import numpy as np


def load_numpy_array_from_txt(path: str, dtype=np.float, delimiter: str=None):
    return np.loadtxt(path, dtype=dtype, delimiter=delimiter)
