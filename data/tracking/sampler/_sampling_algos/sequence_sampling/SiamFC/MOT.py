from datasets.MOT.dataset import MultipleObjectTrackingDatasetSequence_MemoryMapped
from ._algo import do_siamfc_pair_sampling_positive_only, do_siamfc_pair_sampling_negative_only, do_siamfc_pair_sampling
import numpy as np
from data.tracking.sampler._sampling_algos.sequence_sampling.common._dummy_bbox import generate_dummy_bbox_xyxy
from data.operator.bbox.validity import bbox_is_valid
from data.operator.bbox.spatial.utility.half_pixel_offset.image import bounding_box_is_intersect_with_image
from data.tracking.sampler.SiamFC.type import SiamesePairSamplingMethod


def _get_bbox_and_generate_dummy_bbox_if_empty(frame, track_id, rng_engine, reference_bbox):
    if frame.has_object(track_id):
        x_obj_info = frame.get_object_by_id(track_id)
        x_bbox_validity = x_obj_info.get_bounding_box_validity_flag()
        if x_bbox_validity is not None and not x_bbox_validity:
            x_bbox = generate_dummy_bbox_xyxy(frame.get_image_size(), rng_engine, reference_bbox)
        else:
            x_bbox = x_obj_info.get_bounding_box()
    else:
        x_bbox = generate_dummy_bbox_xyxy(frame.get_image_size(), rng_engine, reference_bbox)
    return x_bbox


def _get_positive_frame_image_and_bbox(sequence, index, track_id):
    frame = sequence.get_frame(index)
    frame_image = frame.get_image_path()
    frame_bbox = frame.get_object_by_id(track_id).get_bounding_box()

    assert any(v > 0 for v in frame.get_image_size())
    assert bbox_is_valid(frame_bbox) and bounding_box_is_intersect_with_image(frame_bbox, frame_image.get_image_size())
    return frame_image, frame_bbox


def _get_frame_image_and_bbox(sequence, index, track_id, rng_engine, reference_bbox):
    frame = sequence.get_frame(index)
    frame_image = frame.get_image_path()
    frame_bbox = _get_bbox_and_generate_dummy_bbox_if_empty(frame, track_id, rng_engine, reference_bbox)
    assert any(v > 0 for v in frame.get_image_size())
    assert bbox_is_valid(frame_bbox) and bounding_box_is_intersect_with_image(frame_bbox, frame.get_image_size())
    return frame_image, frame_bbox


def _data_getter(sequence: MultipleObjectTrackingDatasetSequence_MemoryMapped, track_id, index_of_frames, rng_engine: np.random.Generator):
    z_image, z_bbox = _get_positive_frame_image_and_bbox(sequence, index_of_frames[0], track_id)

    if len(index_of_frames) == 1:
        return ((z_image, z_bbox), )

    x_image, x_bbox = _get_frame_image_and_bbox(sequence, index_of_frames[1], track_id, rng_engine, z_bbox)
    return ((z_image, z_bbox), (x_image, x_bbox))


def _sampling_one_track_in_sequence_and_generate_object_visible_mask(sequence: MultipleObjectTrackingDatasetSequence_MemoryMapped, rng_engine: np.random.Generator):
    index_of_track = rng_engine.integers(0, sequence.get_number_of_objects())
    track = sequence.get_object(index_of_track)

    mask = np.zeros(len(sequence), dtype=np.bool_)
    if track.get_all_bounding_box_validity_flag() is not None:
        ind = track.get_all_frame_index()[track.get_all_bounding_box_validity_flag()]
        mask[ind] = True
    else:
        mask[track.get_all_frame_index()] = True
    return mask, track.get_id()


def do_positive_sampling_in_multiple_object_tracking_dataset_sequence(sequence: MultipleObjectTrackingDatasetSequence_MemoryMapped, frame_range: int, sampling_method: SiamesePairSamplingMethod, rng_engine: np.random.Generator):
    mask, track_id = _sampling_one_track_in_sequence_and_generate_object_visible_mask(sequence, rng_engine)

    return _data_getter(sequence, track_id, do_siamfc_pair_sampling_positive_only(len(sequence), frame_range, mask, sampling_method, rng_engine), rng_engine)


def do_negative_sampling_in_multiple_object_tracking_dataset_sequence(sequence: MultipleObjectTrackingDatasetSequence_MemoryMapped, frame_range: int, rng_engine: np.random.Generator):
    mask, track_id = _sampling_one_track_in_sequence_and_generate_object_visible_mask(sequence, rng_engine)

    return _data_getter(sequence, track_id, do_siamfc_pair_sampling_negative_only(len(sequence), frame_range, mask, rng_engine), rng_engine)


def do_sampling_in_multiple_object_tracking_dataset_sequence(sequence: MultipleObjectTrackingDatasetSequence_MemoryMapped, frame_range: int, sampling_method: SiamesePairSamplingMethod, rng_engine: np.random.Generator):
    mask, track_id = _sampling_one_track_in_sequence_and_generate_object_visible_mask(sequence, rng_engine)

    indices, is_positive = do_siamfc_pair_sampling(len(sequence), frame_range, mask, sampling_method, rng_engine)
    return _data_getter(sequence, track_id, indices, rng_engine), is_positive
