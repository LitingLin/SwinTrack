from datasets.MOT.dataset import MultipleObjectTrackingDatasetSequence_MemoryMapped
from ..SiamFC.MOT import _sampling_one_track_in_sequence_and_generate_object_visible_mask, _get_positive_frame_image_and_bbox, _get_frame_image_and_bbox
import numpy as np
from ._algo import do_triplet_sampling_positive_only, do_triplet_sampling_negative_only
from data.tracking.sampler.SiamFC.type import SiamesePairSamplingMethod


def _data_getter(sequence: MultipleObjectTrackingDatasetSequence_MemoryMapped, track_id, index_of_frames, rng_engine: np.random.Generator):
    z_image, z_bbox = _get_positive_frame_image_and_bbox(sequence, index_of_frames[0], track_id)

    if len(index_of_frames) == 1:
        return ((z_image, z_bbox), )

    x_image, x_bbox = _get_frame_image_and_bbox(sequence, index_of_frames[1], track_id, rng_engine, z_bbox)
    if len(index_of_frames) == 2:
        return ((z_image, z_bbox), (x_image, x_bbox))

    assert len(index_of_frames) == 3

    aux_image, aux_bbox = _get_frame_image_and_bbox(sequence, index_of_frames[1], track_id, rng_engine, z_bbox)
    return ((z_image, z_bbox), (x_image, x_bbox), (aux_image, aux_bbox))


def do_triplet_positive_sampling_in_multiple_object_tracking_dataset_sequence(sequence: MultipleObjectTrackingDatasetSequence_MemoryMapped,
                                                                              frame_range: int, sampling_method: SiamesePairSamplingMethod,
                                                                              aux_frame_range: int, aux_sampling_method: SiamesePairSamplingMethod,
                                                                              rng_engine: np.random.Generator):
    mask, track_id = _sampling_one_track_in_sequence_and_generate_object_visible_mask(sequence, rng_engine)

    return _data_getter(sequence, track_id, do_triplet_sampling_positive_only(len(sequence),
                                                                              frame_range, aux_frame_range,
                                                                              mask,
                                                                              sampling_method, aux_sampling_method,
                                                                              rng_engine), rng_engine)


def do_triplet_negative_sampling_in_multiple_object_tracking_dataset_sequence(sequence: MultipleObjectTrackingDatasetSequence_MemoryMapped, frame_range: int, aux_frame_range: int, rng_engine: np.random.Generator):
    mask, track_id = _sampling_one_track_in_sequence_and_generate_object_visible_mask(sequence, rng_engine)

    return _data_getter(sequence, track_id, do_triplet_sampling_negative_only(len(sequence), frame_range, aux_frame_range, mask, rng_engine), rng_engine)
