import torch


def bbox_restrict_in_image_boundary_(bbox: torch.Tensor, image_size):
    torch.clamp_min_(bbox[..., :2], 0.5)
    torch.clamp_max_(bbox[..., 2], image_size[0] - 0.5)
    torch.clamp_max_(bbox[..., 3], image_size[1] - 0.5)
    return bbox
