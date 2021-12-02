from datasets.DET.dataset import DetectionDatasetImage_MemoryMapped
from ._common import _check_bounding_box_validity


class DETImageSequentialSamplerAdaptor:
    def __init__(self, image: DetectionDatasetImage_MemoryMapped, rng_engine):
        self.image = image
        self.index_of_object = rng_engine.integers(0, len(image))

    def move_next(self):
        return False

    def current(self):
        image_path = self.image.get_image_path()
        object_ = self.image[self.index_of_object]
        bounding_box = object_.get_bounding_box()
        bounding_box_validity_flag = object_.get_bounding_box_validity_flag()

        bounding_box = _check_bounding_box_validity(bounding_box, bounding_box_validity_flag, self.image.get_image_size())
        return image_path, bounding_box

    def reset(self):
        pass

    def length(self):
        return 1
