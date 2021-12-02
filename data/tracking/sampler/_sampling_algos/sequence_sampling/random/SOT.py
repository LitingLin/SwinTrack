import numpy as np
from datasets.SOT.dataset import SingleObjectTrackingDatasetSequence_MemoryMapped
from data.tracking.sampler._sampling_algos.sequence_sampling.common._dummy_bbox import generate_dummy_bbox_xyxy


def get_one_random_sample_in_single_object_tracking_dataset_sequence(sequence: SingleObjectTrackingDatasetSequence_MemoryMapped, rng_engine: np.random.Generator):
    index_of_frame = rng_engine.integers(0, len(sequence))
    frame = sequence[index_of_frame]
    bbox_validity = frame.get_bounding_box_validity_flag()
    if bbox_validity is not None and not bbox_validity:
        return frame.get_image_path(), generate_dummy_bbox_xyxy(frame.get_image_size(), rng_engine)
    else:
        return frame.get_image_path(), frame.get_bounding_box()
