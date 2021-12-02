import numpy as np

from .intersection import bbox_compute_intersection_numpy_vectorized
from .area import bbox_compute_area_numpy_vectorized


def bbox_compute_iou_numpy_vectorized(bbox_a: np.ndarray, bbox_b: np.ndarray):
    intersection = bbox_compute_intersection_numpy_vectorized(bbox_a, bbox_b)
    intersection_area = bbox_compute_area_numpy_vectorized(intersection)
    union_area = bbox_compute_area_numpy_vectorized(bbox_a) + bbox_compute_area_numpy_vectorized(bbox_b) - intersection_area
    return intersection_area / union_area
