import numpy as np


def box_xyxy2cxcywh(box: np.ndarray):
    cxcywh_box = np.zeros(4, dtype=np.float_)
    cxcywh_box[0] = (box[0] + box[2]) / 2.
    cxcywh_box[1] = (box[1] + box[3]) / 2.
    cxcywh_box[2] = box[2] - box[0]
    cxcywh_box[3] = box[3] - box[1]
    return cxcywh_box
