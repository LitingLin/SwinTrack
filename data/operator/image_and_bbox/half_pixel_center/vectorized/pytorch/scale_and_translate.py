import torch
import torch.nn.functional
from data.operator.bbox.spatial.vectorized.torch.utility.half_pixel_offset.image import bbox_restrict_in_image_boundary_
from data.operator.bbox.spatial.vectorized.torch.scale_and_translate import bbox_scale_and_translate_vectorized
from data.operator.bbox.spatial.vectorized.torch.validity import bbox_is_valid_vectorized


def torch_scale_and_translate_half_pixel_offset(img, output_size, scale, input_center, output_center,
                                                background_color=None, mode='bilinear', output_img=None):
    """
    Args:
        img (torch.Tensor): (n, c, h, w) or (c, h, w)
        output_size (int, int): (2)
        scale (torch.Tensor): (n, 2) or (2)
        input_center (torch.Tensor): (n, 2) or (2)
        output_center (torch.Tensor): (n, 2) or (2)
        background_color (torch.Tensor | None): (n, c) or (n, 1) or (c)
        mode (str): interpolate algorithm
    Returns:
        (torch.Tensor, torch.Tensor): tuple containing:
            output_image(torch.Tensor): (n, c, h, w) or (c, h, w), curated image
            image_bbox (torch.Tensor): (n, 2) or (2)
    """
    if mode in ('bilinear', 'bicubic'):
        align_corners = True
    else:
        align_corners = None
    assert img.ndim in (3, 4)
    batch_mode = img.ndim == 4
    if not batch_mode:
        img = img.unsqueeze(0)
    if output_img is not None:
        if batch_mode:
            assert output_img.ndim == 4
        else:
            assert output_img.ndim in (3, 4)
            if output_img.ndim == 4:
                assert output_img.shape[0] == 1
            else:
                output_img = output_img.unsqueeze(0)
    n, c, h, w = img.shape
    if background_color is not None:
        if background_color.ndim == 1:
            if output_img is None:
                output_img = background_color.reshape(1, -1, 1, 1).repeat(
                    n, c // background_color.shape[0], output_size[1], output_size[0])
            else:
                output_img[:] = background_color.reshape(1, -1, 1, 1)
        elif background_color.ndim == 2:
            b_n, b_c = background_color.shape
            assert b_n == n
            if output_img is None:
                output_img = background_color.reshape(b_n, b_c, 1, 1).repeat(1, c // b_c, output_size[1], output_size[0])
            else:
                output_img[:] = background_color.reshape(b_n, b_c, 1, 1)
        else:
            raise RuntimeError(f"Incompatible background_color shape")
    else:
        if output_img is None:
            output_img = torch.zeros((n, c, output_size[1], output_size[0]), dtype=img.dtype, device=img.device)

    output_bbox = bbox_scale_and_translate_vectorized(
        torch.tensor((0, 0, w, h), dtype=torch.float64, device=scale.device), scale, input_center, output_center)
    bbox_restrict_in_image_boundary_(output_bbox, output_size)
    input_bbox = bbox_scale_and_translate_vectorized(output_bbox, 1 / scale, output_center, input_center)
    output_bbox = output_bbox.to(torch.int)
    input_bbox = input_bbox.to(torch.int)
    output_bbox_validity = bbox_is_valid_vectorized(output_bbox)

    assert output_bbox.ndim in (1, 2)

    if output_bbox.ndim == 2:
        assert output_bbox.shape[0] == n
        for i_n in range(n):
            if not output_bbox_validity[i_n]:
                continue
            output_img[i_n, :, output_bbox[i_n, 1]: output_bbox[i_n, 3] + 1, output_bbox[i_n, 0]: output_bbox[i_n, 2] + 1] = torch.nn.functional.interpolate(
                img[i_n: i_n + 1, :, input_bbox[i_n, 1]: input_bbox[i_n, 3] + 1, input_bbox[i_n, 0]: input_bbox[i_n, 2] + 1],
                (output_bbox[i_n, 3] - output_bbox[i_n, 1] + 1, output_bbox[i_n, 2] - output_bbox[i_n, 0] + 1),
                mode=mode,
                align_corners=align_corners)
    else:
        if output_bbox_validity:
            for i_n in range(n):
                output_img[i_n, :, output_bbox[1]: output_bbox[3] + 1, output_bbox[0]: output_bbox[2] + 1] = torch.nn.functional.interpolate(
                    img[i_n: i_n + 1, :, input_bbox[1]: input_bbox[3] + 1, input_bbox[0]: input_bbox[2] + 1],
                    (output_bbox[3] - output_bbox[1] + 1, output_bbox[2] - output_bbox[0] + 1),
                    mode=mode,
                    align_corners=align_corners)
    if not batch_mode:
        output_img = output_img.squeeze(0)
    return output_img, output_bbox
