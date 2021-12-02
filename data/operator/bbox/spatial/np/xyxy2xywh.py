import numpy as np


def bbox_xyxy2xywh(box):
    xywh_box = np.empty_like(box)
    xywh_box[..., 0] = box[..., 0]
    xywh_box[..., 1] = box[..., 1]
    xywh_box[..., 2] = box[..., 2] - box[..., 0]
    xywh_box[..., 3] = box[..., 3] - box[..., 1]
    return xywh_box
