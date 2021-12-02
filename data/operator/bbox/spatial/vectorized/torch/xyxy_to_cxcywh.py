import torch


def box_xyxy_to_cxcywh(x: torch.Tensor):
    x0, y0, x1, y1 = x.unbind(-1)
    out = torch.empty_like(x)
    out[..., 0] = (x0 + x1) / 2
    out[..., 1] = (y0 + y1) / 2
    out[..., 2] = (x1 - x0)
    out[..., 3] = (y1 - y0)
    return out
