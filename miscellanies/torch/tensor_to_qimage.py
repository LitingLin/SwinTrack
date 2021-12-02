import torch
from miscellanies.qt_numpy_interop import numpy_rgb888_to_qimage


def torch_tensor_to_qimage(tensor):
    tensor = tensor * 255.
    tensor.clamp_(min=0., max=255.)
    tensor = tensor.to(torch.uint8)
    tensor = tensor.permute(1, 2, 0)
    q_image = numpy_rgb888_to_qimage(tensor.numpy())
    return q_image


def torch_tensor_uint8_to_qimage(tensor):
    tensor = tensor.permute(1, 2, 0)
    q_image = numpy_rgb888_to_qimage(tensor.numpy())
    return q_image
