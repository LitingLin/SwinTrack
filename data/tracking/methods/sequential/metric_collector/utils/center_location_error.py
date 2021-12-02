import numpy as np


def calculate_center_location_error_torch_vectorized(pred_bb, anno_bb, normalized=False):
    pred_center = (pred_bb[:, :2] + pred_bb[:, 2:]) / 2.
    anno_center = (anno_bb[:, :2] + anno_bb[:, 2:]) / 2.

    if normalized:
        anno_wh = anno_bb[:, 2:] - anno_bb[:, :2]
        pred_center = pred_center / anno_wh
        anno_center = anno_center / anno_wh

    err_center = np.sqrt(((pred_center - anno_center)**2).sum(1))
    return err_center
