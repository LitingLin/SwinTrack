import numpy as np
from datasets.DET.dataset import DetectionDatasetImage_MemoryMapped
from data.tracking.sampler._sampling_algos.stateless.random import sampling
from data.tracking.sampler._sampling_algos.sequence_sampling.common._algo import sample_one_positive
from data.tracking.sampler._sampling_algos.sequence_sampling.common._dummy_bbox import generate_dummy_bbox_xyxy
from data.operator.bbox.validity import bbox_is_valid
from data.operator.bbox.spatial.utility.half_pixel_offset.image import bounding_box_is_intersect_with_image


def get_one_random_sample_in_detection_dataset_image(image: DetectionDatasetImage_MemoryMapped, rng_engine: np.random.Generator):
    index_of_object = sampling(len(image), rng_engine)
    object_ = image[index_of_object]
    bbox_validity = object_.get_bounding_box_validity_flag()
    if bbox_validity is not None and not bbox_validity:
        bbox = generate_dummy_bbox_xyxy(image.get_image_size(), rng_engine)
    else:
        bbox = object_.get_bounding_box()
    assert any(v > 0 for v in image.get_image_size())
    assert bbox_is_valid(bbox) and bounding_box_is_intersect_with_image(bbox, image.get_image_size())
    return image.get_image_path(), bbox


def do_sampling_in_detection_dataset_image(image: DetectionDatasetImage_MemoryMapped, rng_engine: np.random.Generator):
    index_of_object = sample_one_positive(len(image), image.get_all_bounding_box_validity_flag(), rng_engine)
    object_ = image[index_of_object]
    bbox = object_.get_bounding_box()
    assert any(v > 0 for v in image.get_image_size())
    assert bbox_is_valid(bbox) and bounding_box_is_intersect_with_image(bbox, image.get_image_size())
    return image.get_image_path(), object_.get_bounding_box()
