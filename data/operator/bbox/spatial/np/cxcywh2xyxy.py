import numpy as np


def box_cxcywh2xyxy(box: np.ndarray):
    xyxy_box = np.zeros(4, dtype=np.float_)
    w_2 = box[2] / 2
    h_2 = box[3] / 2
    xyxy_box[0] = box[0] - w_2
    xyxy_box[1] = box[1] - h_2
    xyxy_box[2] = box[0] + w_2
    xyxy_box[3] = box[1] + h_2
    return xyxy_box
