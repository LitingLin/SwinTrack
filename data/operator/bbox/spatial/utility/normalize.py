class BoundingBoxNormalizationHelper:
    def __init__(self, interval, range_):
        assert interval in ('[)', '[]')
        self.right_open = (interval == '[)')
        assert range_[1] > range_[0]
        self.scale = range_[1] - range_[0]
        self.offset = range_[0]

    def normalize(self, bbox, image_size):
        if self.right_open:
            bbox = tuple(v / image_size[i % 2] for i, v in enumerate(bbox))
        else:
            bbox = tuple(v / (image_size[i % 2] - 1) for i, v in enumerate(bbox))
        bbox = tuple(v * self.scale + self.offset for v in bbox)
        return bbox

    def denormalize(self, bbox, image_size):
        bbox = tuple((v - self.offset) / self.scale for v in bbox)
        if self.right_open:
            bbox = tuple(v * image_size[i % 2] for i, v in enumerate(bbox))
        else:
            bbox = tuple(v * (image_size[i % 2] - 1) for i, v in enumerate(bbox))
        return bbox
