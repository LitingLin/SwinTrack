# Copyright (c) OpenMMLab. All rights reserved.
# https://github.com/open-mmlab/mmdetection/blob/master/mmdet/models/losses/iou_loss.py
from core.operator.iou.iou2d_calculator import bbox_overlaps


def iou(pred, target, eps=1e-6):
    ious = bbox_overlaps(pred, target, is_aligned=True).clamp(min=eps)
    return ious
