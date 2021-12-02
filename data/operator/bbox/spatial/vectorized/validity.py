import numpy as np


def bbox_is_valid_vectorized(bbox):
    validity = bbox[:, :2] < bbox[:, 2:]
    return np.logical_and(validity[:, 0], validity[:, 1])
