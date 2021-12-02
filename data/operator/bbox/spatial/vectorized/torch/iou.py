import torch
from .intersection import bbox_compute_intersection_vectorized


def _compute_area(bbox):
    return torch.prod(bbox[..., 2:] - bbox[..., :2], dim=-1)


def bbox_compute_iou_vectorized(bbox_a: torch.Tensor, bbox_b: torch.Tensor):
    intersection = bbox_compute_intersection_vectorized(bbox_a, bbox_b)
    intersection_area = _compute_area(intersection)
    union_area = _compute_area(bbox_a) + _compute_area(bbox_b) - intersection_area
    return intersection_area / union_area
