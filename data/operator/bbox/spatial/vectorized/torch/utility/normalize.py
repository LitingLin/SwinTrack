import torch


class BoundingBoxNormalizationHelper:
    def __init__(self, interval, range_):
        assert interval in ('[)', '[]')
        self.right_open = (interval == '[)')
        assert range_[1] > range_[0]
        self.scale = range_[1] - range_[0]
        self.offset = range_[0]

    def normalize_(self, bbox: torch.Tensor, image_size):
        if self.right_open:
            bbox[..., ::2] /= image_size[0]
            bbox[..., 1::2] /= image_size[1]
        else:
            bbox[..., ::2] /= (image_size[0] - 1)
            bbox[..., 1::2] /= (image_size[1] - 1)
        bbox *= self.scale
        bbox += self.offset
        return bbox

    def normalize(self, bbox: torch.Tensor, image_size):
        return self.normalize_(bbox.clone(), image_size)

    def denormalize_(self, bbox: torch.Tensor, image_size):
        bbox -= self.offset
        bbox /= self.scale
        if self.right_open:
            bbox[..., ::2] *= image_size[0]
            bbox[..., 1::2] *= image_size[1]
        else:
            bbox[..., ::2] *= (image_size[0] - 1)
            bbox[..., 1::2] *= (image_size[1] - 1)
        return bbox

    def denormalize(self, bbox: torch.Tensor, image_size):
        return self.denormalize_(bbox.clone(), image_size)
