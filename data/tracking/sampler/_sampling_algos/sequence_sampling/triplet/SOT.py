import numpy as np
from ._algo import do_triplet_sampling_positive_only, do_triplet_sampling_negative_only
from ..SiamFC.SOT import _get_positive_frame_image_and_bbox, _get_frame_image_and_bbox


def _data_getter(sequence, indices, rng_engine: np.random.Generator):
    z_image, z_bbox = _get_positive_frame_image_and_bbox(sequence, indices[0])
    if len(indices) == 1:
        return ((z_image, z_bbox), )

    x_image, x_bbox = _get_frame_image_and_bbox(sequence, indices[1], rng_engine, z_bbox)

    if len(indices) == 2:
        return ((z_image, z_bbox), (x_image, x_bbox))

    assert len(indices) == 3
    aux_image, aux_bbox = _get_frame_image_and_bbox(sequence, indices[2], rng_engine, z_bbox)
    return ((z_image, z_bbox), (x_image, x_bbox), (aux_image, aux_bbox))


def do_triplet_positive_sampling_in_single_object_tracking_dataset_sequence(sequence, frame_range: int, sampling_method, aux_frame_range: int, aux_sampling_method, rng_engine: np.random.Generator):
    return _data_getter(sequence, do_triplet_sampling_positive_only(len(sequence), frame_range, aux_frame_range,
                                                                    sequence.get_all_bounding_box_validity_flag(),
                                                                    sampling_method, aux_sampling_method, rng_engine), rng_engine)


def do_triplet_negative_sampling_in_single_object_tracking_dataset_sequence(sequence, frame_range: int, aux_frame_range: int, rng_engine: np.random.Generator):
    return _data_getter(sequence, do_triplet_sampling_negative_only(len(sequence), frame_range, aux_frame_range, sequence.get_all_bounding_box_validity_flag(), rng_engine), rng_engine)
