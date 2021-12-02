import torch


def bbox_scale_and_translate_vectorized(bbox, scale, input_center, output_center):
    """
    (i - input_center) * scale = o - output_center
    Args:
        bbox (torch.Tensor): (n, 4)
        scale (torch.Tensor): (n, 2)
        input_center (torch.Tensor): (n, 2)
        output_center (torch.Tensor): (n, 2)
    Returns:
        torch.Tensor: scaled torch tensor, (n, 4)
    """
    out_bbox = torch.empty_like(bbox)
    out_bbox[..., ::2] = bbox[..., ::2] - input_center[..., (0, )]
    out_bbox[..., ::2] *= scale[..., (0, )]
    out_bbox[..., ::2] += output_center[..., (0, )]

    out_bbox[..., 1::2] = bbox[..., 1::2] - input_center[..., (1, )]
    out_bbox[..., 1::2] *= scale[..., (1, )]
    out_bbox[..., 1::2] += output_center[..., (1, )]
    return out_bbox
