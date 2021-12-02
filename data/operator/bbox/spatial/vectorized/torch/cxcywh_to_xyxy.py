import torch


def box_cxcywh_to_xyxy(x: torch.Tensor):
    x_c, y_c, w, h = x.unbind(-1)
    out = torch.empty_like(x)
    half_w = 0.5 * w
    half_h = 0.5 * h
    out[..., 0] = (x_c - half_w)
    out[..., 1] = (y_c - half_h)
    out[..., 2] = (x_c + half_w)
    out[..., 3] = (y_c + half_h)
    return out
