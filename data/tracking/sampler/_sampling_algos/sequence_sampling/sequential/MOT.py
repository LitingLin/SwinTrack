from datasets.MOT.dataset import MultipleObjectTrackingDatasetSequence_MemoryMapped
from ._common import _check_bounding_box_validity
import numpy as np


class MOTSequenceSequentialSampler:
    def __init__(self, sequence: MultipleObjectTrackingDatasetSequence_MemoryMapped, rng_engine):
        assert sequence.get_number_of_objects() > 0
        index_of_object = rng_engine.integers(0, sequence.get_number_of_objects())
        sequence_object = sequence.get_object(index_of_object)

        self.sequence = sequence
        self.object_id = sequence_object.get_id()
        self.frame_indices = sequence_object.get_all_frame_index()
        assert len(self.frame_indices) > 0

        self.begin_offset = self.frame_indices[0]
        self.end_offset = self.frame_indices[-1]

        self.index = self.begin_offset

    def get_name(self):
        return f'{self.sequence.get_name()}-track{self.object_id}'

    def move_next(self):
        if self.index + 1 > self.end_offset:
            return False
        self.index += 1
        return True

    def current(self):
        index_of_object_track = np.searchsorted(self.frame_indices, self.index)
        frame = self.sequence.get_frame(self.index)
        image_path = frame.get_image_path()
        assert any(v > 0 for v in frame.get_image_size())
        if self.frame_indices[index_of_object_track] != self.index:
            bounding_box = None
        else:
            frame_object = frame.get_object_by_id(self.object_id)
            bounding_box = frame_object.get_bounding_box()
            bounding_box_validity_flag = frame_object.get_bounding_box_validity_flag()

            bounding_box = _check_bounding_box_validity(bounding_box, bounding_box_validity_flag, frame.get_image_size())
        return image_path, bounding_box

    def reset(self):
        self.index = self.begin_offset

    def length(self):
        return self.end_offset - self.begin_offset + 1
