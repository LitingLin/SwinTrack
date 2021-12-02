import torch


def get_image_mean_nchw(image, out=None):
    """
    Args:
        image(torch.Tensor): (n, c, h, w)
    """
    return torch.mean(image, dim=(2, 3), out=out)


def get_image_mean_chw(image, out=None):
    """
    Args:
        image(torch.Tensor): (c, h, w)
    """
    return torch.mean(image, dim=(1, 2), out=out)


def get_image_mean_hw(image, out=None):
    """
    Args:
        image(torch.Tensor): (h, w)
    """
    return torch.mean(image, out=out)


def get_image_mean(image, out=None):
    assert image.ndim in (2, 3, 4)
    if image.ndim == 2:
        return torch.mean(image, out=out)
    elif image.ndim == 3:
        return get_image_mean_chw(image, out)
    else:
        return get_image_mean_nchw(image, out)
