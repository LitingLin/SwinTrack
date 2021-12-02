import ctypes
import numpy as np


def memmove_to_ctypes(src: np.ndarray, dst):
    assert dst.nbytes == ctypes.sizeof(dst)
    src = np.ascontiguousarray(src)
    ctypes.memmove(ctypes.byref(dst), src.ctypes.data, src.nbytes)
