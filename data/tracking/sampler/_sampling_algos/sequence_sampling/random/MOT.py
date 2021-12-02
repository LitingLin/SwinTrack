import numpy as np
from data.tracking.sampler._sampling_algos.sequence_sampling.common._dummy_bbox import generate_dummy_bbox_xyxy
from datasets.MOT.dataset import MultipleObjectTrackingDatasetSequence_MemoryMapped


def get_one_random_sample_in_multiple_object_tracking_dataset_sequence(sequence: MultipleObjectTrackingDatasetSequence_MemoryMapped, rng_engine: np.random.Generator):
    index_of_frame = rng_engine.integers(0, sequence.get_number_of_frames())
    frame = sequence[index_of_frame]
    if len(frame) == 0:
        return frame.get_image_path(), generate_dummy_bbox_xyxy(frame.get_image_size(), rng_engine)
    else:
        index_of_frame_object = rng_engine.integers(0, len(frame))
        frame_object = frame[index_of_frame_object]
        bbox_validity = frame_object.get_bounding_box_validity_flag()
        if bbox_validity is not None and not bbox_validity:
            return frame.get_image_path(), generate_dummy_bbox_xyxy(frame.get_image_size(), rng_engine)
        else:
            return frame.get_image_path(), frame_object.get_bounding_box()
