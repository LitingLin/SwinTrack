from datasets.SOT.dataset import SingleObjectTrackingDatasetSequence_MemoryMapped
from ._algo import do_siamfc_pair_sampling_positive_only, do_siamfc_pair_sampling_negative_only, do_siamfc_pair_sampling
from data.tracking.sampler._sampling_algos.sequence_sampling.common._dummy_bbox import generate_dummy_bbox_xyxy
import numpy as np
from data.operator.bbox.validity import bbox_is_valid
from data.operator.bbox.spatial.utility.half_pixel_offset.image import bounding_box_is_intersect_with_image
from data.tracking.sampler.SiamFC.type import SiamesePairSamplingMethod


def _get_positive_frame_image_and_bbox(sequence, index):
    frame = sequence[index]
    frame_image = frame.get_image_path()
    frame_bbox = frame.get_bounding_box()
    assert any(v > 0 for v in frame.get_image_size())
    assert bbox_is_valid(frame_bbox) and bounding_box_is_intersect_with_image(frame_bbox, frame.get_image_size())
    return frame_image, frame_bbox


def _get_frame_image_and_bbox(sequence, index, rng_engine, reference_bbox):
    frame = sequence[index]
    frame_image = frame.get_image_path()
    frame_bbox = frame.get_bounding_box()
    frame_bbox_validity = frame.get_bounding_box_validity_flag()
    if frame_bbox_validity is not None and not frame_bbox_validity:
        frame_bbox = generate_dummy_bbox_xyxy(frame.get_image_size(), rng_engine, reference_bbox)
    else:
        assert bbox_is_valid(frame_bbox) and bounding_box_is_intersect_with_image(frame_bbox, frame.get_image_size())
    assert any(v > 0 for v in frame.get_image_size())
    return frame_image, frame_bbox


def _data_getter(sequence, indices, rng_engine: np.random.Generator):
    z_image, z_bbox = _get_positive_frame_image_and_bbox(sequence, indices[0])
    if len(indices) == 1:
        return ((z_image, z_bbox), )

    x_image, x_bbox = _get_frame_image_and_bbox(sequence, indices[1], rng_engine, z_bbox)
    return ((z_image, z_bbox), (x_image, x_bbox))


def do_positive_sampling_in_single_object_tracking_dataset_sequence(sequence: SingleObjectTrackingDatasetSequence_MemoryMapped, frame_range: int, sampling_method: SiamesePairSamplingMethod, rng_engine: np.random.Generator):
    return _data_getter(sequence, do_siamfc_pair_sampling_positive_only(len(sequence), frame_range, sequence.get_all_bounding_box_validity_flag(), sampling_method, rng_engine), rng_engine)


def do_negative_sampling_in_single_object_tracking_dataset_sequence(sequence: SingleObjectTrackingDatasetSequence_MemoryMapped, frame_range: int, rng_engine: np.random.Generator):
    return _data_getter(sequence, do_siamfc_pair_sampling_negative_only(len(sequence), frame_range, sequence.get_all_bounding_box_validity_flag(), rng_engine), rng_engine)


def do_sampling_in_single_object_tracking_dataset_sequence(sequence: SingleObjectTrackingDatasetSequence_MemoryMapped, frame_range: int, sampling_method: SiamesePairSamplingMethod, rng_engine: np.random.Generator):
    indices, is_positive = do_siamfc_pair_sampling(len(sequence), frame_range, sequence.get_all_bounding_box_validity_flag(), sampling_method, rng_engine)
    return _data_getter(sequence, indices, rng_engine), is_positive
