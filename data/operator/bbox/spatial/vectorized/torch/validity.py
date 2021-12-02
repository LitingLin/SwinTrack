import torch


def bbox_is_valid_vectorized(bbox: torch.Tensor):
    validity = bbox[..., :2] < bbox[..., 2:]
    return torch.logical_and(validity[..., 0], validity[..., 1])
