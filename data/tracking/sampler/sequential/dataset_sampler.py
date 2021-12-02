import numpy as np

from data.tracking.sampler._sampling_algos.sequence_sampling.sequential.SOT import SOTSequenceSequentialSampler
from data.tracking.sampler._sampling_algos.sequence_sampling.sequential.DET import DETImageSequentialSamplerAdaptor
from data.tracking.sampler._sampling_algos.sequence_sampling.sequential.MOT import MOTSequenceSequentialSampler

from datasets.SOT.dataset import SingleObjectTrackingDatasetSequence_MemoryMapped
from datasets.MOT.dataset import MultipleObjectTrackingDatasetSequence_MemoryMapped
from datasets.DET.dataset import DetectionDatasetImage_MemoryMapped

import uuid


'''
RandomSampling support DET SOT MOT
RunThrough support SOT only
'''


class _SequentialDatasetWrapper:
    def __init__(self, sequence, rng_engine):
        if isinstance(sequence, SingleObjectTrackingDatasetSequence_MemoryMapped):
            sequence = SOTSequenceSequentialSampler(sequence)
            assert sequence.length() > 1
        elif isinstance(sequence, MultipleObjectTrackingDatasetSequence_MemoryMapped):
            sequence = MOTSequenceSequentialSampler(sequence, rng_engine)
            assert sequence.length() > 1
        elif isinstance(sequence, DetectionDatasetImage_MemoryMapped):
            sequence = DETImageSequentialSamplerAdaptor(sequence, rng_engine)
        else:
            raise NotImplementedError
        self.sequence = sequence

        image_path, bbox = self.sequence.current()
        self.template_image_path = image_path
        self.template_object_bbox = bbox.astype(np.float)
        self.position = 1
        if not isinstance(sequence, DETImageSequentialSamplerAdaptor):
            assert self.sequence.move_next()
        assert bbox is not None

    def get_name(self):
        if isinstance(self.sequence, DetectionDatasetImage_MemoryMapped):
            return None
        return self.sequence.get_name()

    def get_length(self):
        if isinstance(self.sequence, DetectionDatasetImage_MemoryMapped):
            return 2
        else:
            return self.sequence.length()

    def __len__(self):
        return self.get_length()

    def move_next(self):
        is_success = self.sequence.move_next()
        if is_success:
            self.position += 1
        return is_success

    def get_position(self):
        return self.position

    def current(self):
        image_path, bbox = self.sequence.current()
        if bbox is not None:
            bbox = bbox.astype(np.float)
        return image_path, bbox

    def get_template(self):
        return self.template_image_path, self.template_object_bbox


class SequentialDatasetSampler:
    def __init__(self, sequence_picker, datasets):
        self.sequence_picker = sequence_picker

        self.datasets = datasets

        self.dataset_unique_id = None
        self.sequence = None
        self.uuid = None

    def get_next(self, rng_engine):
        if self.sequence is None or not self.sequence.move_next():
            sequence_picking_result = self.sequence_picker.get_next()
            if sequence_picking_result is None:
                self.sequence = None
                return None
            index_of_dataset, index_of_sequence = sequence_picking_result
            dataset = self.datasets[index_of_dataset]
            sequence = self.datasets[index_of_dataset][index_of_sequence]
            self.dataset_unique_id = dataset.get_unique_id()
            self.sequence = _SequentialDatasetWrapper(sequence, rng_engine)
            self.uuid = uuid.uuid1()

        return self.uuid, self.sequence.get_name(), self.dataset_unique_id, self.sequence.get_position(), self.sequence.get_length(), self.sequence.get_template(), self.sequence.current()


class SequentialDataset:
    def __init__(self, samplers, post_processor):
        self.samplers = samplers
        self.post_processor = post_processor
        self.index_iter = iter(range(len(self.samplers)))

    def get_next(self, rng_engine):
        while True:
            try:
                index = next(self.index_iter)
                break
            except StopIteration:
                self.index_iter = iter(range(len(self.samplers)))
        return self.post_processor(index, self.samplers[index].get_next(rng_engine))
