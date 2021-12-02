import torch
from typing import Sequence
import math


def generate_sine_position_encoding(batch, N_feature_dim: int, pos_indices: Sequence[torch.Tensor], normalized=True,
                                    scale=math.pi * 2, temperature=10000, eps=torch.finfo(torch.float32).eps):
    assert len(pos_indices) > 0

    step_feat_len = N_feature_dim // len(pos_indices)
    last_step_feat_len = N_feature_dim - step_feat_len * (len(pos_indices) - 1)

    shape = tuple(len(direction_index) for direction_index in pos_indices)

    device = pos_indices[0].device
    dtype = pos_indices[0].dtype

    position_encoding = torch.empty((batch, *shape, N_feature_dim), device=device, dtype=dtype)

    for index, pos_index in enumerate(pos_indices):
        feat_dim_begin = step_feat_len * index
        if index == len(pos_indices) - 1:
            step_feat_len = last_step_feat_len
        if normalized:
            pos_index /= (pos_index[-1] + eps)
            pos_index *= scale
        step_feat_indices = torch.arange(step_feat_len, dtype=torch.float, device=device)
        step_feat_indices = temperature ** (
                2 * (torch.div(step_feat_indices, 2, rounding_mode='trunc')) / step_feat_len)
        encoding = pos_index.unsqueeze(-1) / step_feat_indices
        encoding[:, 0::2].sin_()
        encoding[:, 1::2].cos_()
        encoding_shape = [1] * (len(shape) + 2)
        encoding_shape[index + 1] = len(pos_index)
        encoding_shape[-1] = -1
        encoding = encoding.view(*encoding_shape)

        slice_object_list = [slice(None)] * (len(shape) + 1)
        slice_object_list.append(slice(feat_dim_begin, feat_dim_begin + step_feat_len))

        position_encoding[slice_object_list] = encoding.to(dtype)
    return position_encoding


def generate_2d_sine_position_encoding(batch, h, w, dim, device=torch.device('cpu'), dtype=torch.float,
                                       normalized=True, scale=math.pi * 2, temperature=10000,
                                       eps=torch.finfo(torch.float32).eps):
    return generate_sine_position_encoding(batch, dim,
                                           (torch.arange(1, h + 1, dtype=dtype, device=device),
                                            torch.arange(1, w + 1, dtype=dtype, device=device)),
                                           normalized, scale, temperature, eps)


def generate_2d_sine_position_encoding_with_index(index, batch, h, w, dim, device=torch.device('cpu'), dtype=torch.float,
                                               normalized=True, scale=math.pi * 2, temperature=10000,
                                               eps=torch.finfo(torch.float32).eps):
    return generate_sine_position_encoding(batch, dim,
                                           (torch.arange(1, h + 1, dtype=dtype, device=device),
                                            torch.arange(1, w + 1, dtype=dtype, device=device),
                                            torch.tensor([index + 1], dtype=dtype, device=device)),
                                           normalized, scale, temperature, eps).view(batch, h, w, dim)
