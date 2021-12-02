import numpy as np


def bounding_box_fit_in_image_boundary_(bbox_: np.ndarray, image_size):
    min_x = 0
    min_y = 0
    max_x = image_size[0] - 1
    max_y = image_size[1] - 1
    bbox_[0] = np.minimum(np.maximum(bbox_[0], min_x), max_x)
    bbox_[1] = np.minimum(np.maximum(bbox_[1], min_y), max_y)
    bbox_[2] = np.minimum(np.maximum(bbox_[2], min_x), max_x)
    bbox_[3] = np.minimum(np.maximum(bbox_[3], min_y), max_y)
