from datasets.SOT.dataset import SingleObjectTrackingDatasetSequence_MemoryMapped
from ._common import _check_bounding_box_validity


class SOTSequenceSequentialSampler:
    def __init__(self, sequence: SingleObjectTrackingDatasetSequence_MemoryMapped):
        assert len(sequence) > 0
        self.sequence = sequence
        self.index = 0

    def get_name(self):
        return self.sequence.get_name()

    def move_next(self):
        if self.index + 1 >= len(self.sequence):
            return False

        self.index += 1
        return True

    def current(self):
        frame = self.sequence[self.index]
        assert any(v > 0 for v in frame.get_image_size())
        image_path = frame.get_image_path()
        bounding_box = frame.get_bounding_box()
        bounding_box_validity_flag = frame.get_bounding_box_validity_flag()

        bounding_box = _check_bounding_box_validity(bounding_box, bounding_box_validity_flag, frame.get_image_size())
        return image_path, bounding_box

    def reset(self):
        self.index = 0

    def length(self):
        return len(self.sequence)
