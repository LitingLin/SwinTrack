import numpy as np


def bbox_compute_area_numpy_vectorized(bbox: np.ndarray):
    return np.prod(bbox[:, 2:] - bbox[:, :2], axis=1)
