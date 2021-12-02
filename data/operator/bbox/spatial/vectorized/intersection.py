import numpy as np


def bbox_compute_intersection_numpy_vectorized(bbox_a: np.ndarray, bbox_b: np.ndarray):
    intersection = np.concatenate((np.maximum(bbox_a[..., :2], bbox_b[..., :2]), np.minimum(bbox_a[..., 2:], bbox_b[..., 2:])), axis=-1)
    intersection[(intersection[..., 2] - intersection[..., 0]) <= 0, :] = 0
    intersection[(intersection[..., 3] - intersection[..., 1]) <= 0, :] = 0
    return intersection
