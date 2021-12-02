import numpy as np


def try_get_int_array(array: np.ndarray):
    int_array = array.astype(np.int)
    if (int_array == array).all():
        return int_array
    else:
        return array
