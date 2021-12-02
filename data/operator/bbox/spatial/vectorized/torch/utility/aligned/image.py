import torch


def bbox_restrict_in_image_boundary_(bbox: torch.Tensor, image_size):
    torch.clamp_min_(bbox[..., :2], 0)
    torch.clamp_max_(bbox[..., 2], image_size[0] - 1)
    torch.clamp_max_(bbox[..., 3], image_size[1] - 1)
    return bbox
