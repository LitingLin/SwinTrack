import numpy as np
from datasets.SOT.dataset import SingleObjectTrackingDataset_MemoryMapped
from datasets.MOT.dataset import MultipleObjectTrackingDataset_MemoryMapped
from datasets.DET.dataset import DetectionDataset_MemoryMapped

from data.tracking.sampler._sampling_algos.sequence_sampling.random.DET import \
    get_one_random_sample_in_detection_dataset_image, \
    do_sampling_in_detection_dataset_image
from data.tracking.sampler._sampling_algos.sequence_sampling.random.SOT import \
    get_one_random_sample_in_single_object_tracking_dataset_sequence
from data.tracking.sampler._sampling_algos.sequence_sampling.random.MOT import \
    get_one_random_sample_in_multiple_object_tracking_dataset_sequence

from data.tracking.sampler._sampling_algos.sequence_sampling.SiamFC.SOT import \
    do_positive_sampling_in_single_object_tracking_dataset_sequence, \
    do_sampling_in_single_object_tracking_dataset_sequence, \
    do_negative_sampling_in_single_object_tracking_dataset_sequence
from data.tracking.sampler._sampling_algos.sequence_sampling.SiamFC.MOT import \
    do_positive_sampling_in_multiple_object_tracking_dataset_sequence, \
    do_sampling_in_multiple_object_tracking_dataset_sequence, \
    do_negative_sampling_in_multiple_object_tracking_dataset_sequence

from data.tracking.sampler.SiamFC.type import SiamesePairSamplingMethod


class _BaseSOTTrackingSiameseIterableDatasetSampler:
    def __init__(self, sequence_picker, datasets, negative_sample_ratio, enforce_fine_positive_sample,
                 sampling_method: SiamesePairSamplingMethod, adaptive_frame_range,
                 negative_sample_random_picking_ratio = 0.,
                 default_frame_range=100, datasets_sampling_parameters=None, datasets_sampling_weight=None,
                 data_processor=None):
        self.sequence_picker = sequence_picker

        self.datasets = datasets

        self.dataset_lengths = [len(dataset) for dataset in datasets]
        self.datasets_sampling_weight = datasets_sampling_weight
        self.enforce_fine_positive_sample = enforce_fine_positive_sample
        self.sampling_method = sampling_method
        self.adaptive_frame_range = adaptive_frame_range
        self.default_frame_range = default_frame_range
        self.negative_sample_ratio = negative_sample_ratio
        self.data_processor = data_processor
        self.datasets_sampling_parameters = datasets_sampling_parameters
        self.negative_sample_random_picking_ratio = negative_sample_random_picking_ratio

        self.current_index_of_dataset = None
        self.current_index_of_sequence = None
        self.current_is_sampling_positive_sample = None

    def _update_sample_type(self, rng_engine: np.random.Generator):
        if self.negative_sample_ratio == 0:
            is_negative = False
        else:
            is_negative = rng_engine.random() < self.negative_sample_ratio
        self.current_is_sampling_positive_sample = not is_negative

    def _pick_random_object_as_negative_sample(self, rng_engine: np.random.Generator):
        index_of_dataset = rng_engine.choice(np.arange(len(self.datasets)), p=self.datasets_sampling_weight)
        dataset = self.datasets[index_of_dataset]
        index_of_sequence = rng_engine.integers(0, len(dataset))
        sequence = dataset[index_of_sequence]
        if isinstance(dataset, DetectionDataset_MemoryMapped):
            data = get_one_random_sample_in_detection_dataset_image(sequence, rng_engine)
        elif isinstance(dataset, SingleObjectTrackingDataset_MemoryMapped):
            data = get_one_random_sample_in_single_object_tracking_dataset_sequence(sequence, rng_engine)
        elif isinstance(dataset, MultipleObjectTrackingDataset_MemoryMapped):
            data = get_one_random_sample_in_multiple_object_tracking_dataset_sequence(sequence, rng_engine)
        else:
            raise NotImplementedError
        return data

    def do_sampling(self, rng_engine: np.random.Generator):
        dataset = self.datasets[self.current_index_of_dataset]
        sequence = dataset[self.current_index_of_sequence]

        frame_range = self.default_frame_range
        if self.datasets_sampling_parameters is not None:
            sampling_parameter = self.datasets_sampling_parameters[self.current_index_of_dataset]
            if 'frame_range' in sampling_parameter:
                frame_range = sampling_parameter['frame_range']
        if self.adaptive_frame_range:
            if isinstance(dataset,
                          (SingleObjectTrackingDataset_MemoryMapped, MultipleObjectTrackingDataset_MemoryMapped)):
                if sequence.has_fps():
                    fps = sequence.get_fps()
                    frame_range = max(int(round(fps / 30 * frame_range)), 1)

        if self.current_is_sampling_positive_sample:
            if isinstance(dataset, DetectionDataset_MemoryMapped):
                z_image, z_bbox = do_sampling_in_detection_dataset_image(sequence, rng_engine)
                data = (z_image, z_bbox, z_image, z_bbox, True)
            elif isinstance(dataset,
                            (SingleObjectTrackingDataset_MemoryMapped, MultipleObjectTrackingDataset_MemoryMapped)):
                if self.enforce_fine_positive_sample:
                    if isinstance(dataset, SingleObjectTrackingDataset_MemoryMapped):
                        sampled_data = \
                            do_positive_sampling_in_single_object_tracking_dataset_sequence(sequence, frame_range,
                                                                                            self.sampling_method,
                                                                                            rng_engine)
                    else:
                        sampled_data = \
                            do_positive_sampling_in_multiple_object_tracking_dataset_sequence(sequence, frame_range,
                                                                                              self.sampling_method,
                                                                                              rng_engine)
                    if len(sampled_data) != 2:
                        return None
                    data = (sampled_data[0][0], sampled_data[0][1], sampled_data[1][0], sampled_data[1][1], True)
                else:
                    if isinstance(dataset, SingleObjectTrackingDataset_MemoryMapped):
                        sampled_data, is_positive = \
                            do_sampling_in_single_object_tracking_dataset_sequence(sequence, frame_range,
                                                                                   self.sampling_method, rng_engine)
                    else:
                        sampled_data, is_positive = \
                            do_sampling_in_multiple_object_tracking_dataset_sequence(sequence, frame_range,
                                                                                     self.sampling_method, rng_engine)
                    if is_positive == 0:
                        data = (sampled_data[0][0], sampled_data[0][1], sampled_data[0][0], sampled_data[0][1], True)
                    else:
                        data = (sampled_data[0][0], sampled_data[0][1], sampled_data[1][0], sampled_data[1][1],
                                is_positive == 1)
            else:
                raise NotImplementedError
        else:
            if isinstance(dataset, DetectionDataset_MemoryMapped):
                z_image, z_bbox = do_sampling_in_detection_dataset_image(sequence, rng_engine)
                x_image, x_bbox = self._pick_random_object_as_negative_sample(rng_engine)
                data = (z_image, z_bbox, x_image, x_bbox, False)
            elif isinstance(dataset,
                            (SingleObjectTrackingDataset_MemoryMapped, MultipleObjectTrackingDataset_MemoryMapped)):
                if isinstance(dataset, SingleObjectTrackingDataset_MemoryMapped):
                    sampled_data = \
                        do_negative_sampling_in_single_object_tracking_dataset_sequence(sequence, frame_range,
                                                                                        rng_engine)
                else:
                    sampled_data = \
                        do_negative_sampling_in_multiple_object_tracking_dataset_sequence(sequence, frame_range,
                                                                                          rng_engine)
                if len(sampled_data) == 1 or (self.negative_sample_random_picking_ratio > 0. and rng_engine.random() < self.negative_sample_random_picking_ratio):
                    x_image, x_bbox = self._pick_random_object_as_negative_sample(rng_engine)
                    data = (sampled_data[0][0], sampled_data[0][1], x_image, x_bbox, False)
                else:
                    data = (sampled_data[0][0], sampled_data[0][1], sampled_data[1][0], sampled_data[1][1], False)
            else:
                raise NotImplementedError
        if self.data_processor is not None:
            data = self.data_processor(*data)
        return data


class SOTTrackingSiameseForwardIteratorDatasetSampler(_BaseSOTTrackingSiameseIterableDatasetSampler):
    def __init__(self, forward_only_sequence_picker, datasets, negative_sample_ratio, enforce_fine_positive_sample,
                 sampling_method: SiamesePairSamplingMethod, adaptive_frame_range,
                 negative_sample_random_picking_ratio = 0.,
                 default_frame_range=100, datasets_sampling_parameters=None, datasets_sampling_weight=None,
                 data_processor=None):
        super(SOTTrackingSiameseForwardIteratorDatasetSampler, self).__init__(forward_only_sequence_picker,
                                                                              datasets, negative_sample_ratio,
                                                                              enforce_fine_positive_sample,
                                                                              sampling_method,
                                                                              adaptive_frame_range,
                                                                              negative_sample_random_picking_ratio,
                                                                              default_frame_range,
                                                                              datasets_sampling_parameters,
                                                                              datasets_sampling_weight,
                                                                              data_processor)

    def get_next(self, rng_engine: np.random.Generator):
        self.move_next(rng_engine)

        while True:
            data = self.do_sampling(rng_engine)
            if data is None:
                self._move_next()
            else:
                return data

    def _move_next(self):
        index_of_dataset, index_of_sequence = self.sequence_picker.get_next()
        self.current_index_of_dataset = index_of_dataset
        self.current_index_of_sequence = index_of_sequence

    def move_next(self, rng_engine: np.random.Generator):
        self._move_next()
        self._update_sample_type(rng_engine)


class SOTTrackingSiameseRandomAccessIteratorDatasetSampler(_BaseSOTTrackingSiameseIterableDatasetSampler):
    def __init__(self, random_accessible_sequence_picker, datasets, negative_sample_ratio, enforce_fine_positive_sample,
                 sampling_method: SiamesePairSamplingMethod, adaptive_frame_range,
                 negative_sample_random_picking_ratio = 0.,
                 default_frame_range=100, datasets_sampling_parameters=None, datasets_sampling_weight=None,
                 data_processor=None):
        super(SOTTrackingSiameseRandomAccessIteratorDatasetSampler, self).__init__(random_accessible_sequence_picker,
                                                                                   datasets, negative_sample_ratio,
                                                                                   enforce_fine_positive_sample,
                                                                                   sampling_method,
                                                                                   adaptive_frame_range,
                                                                                   negative_sample_random_picking_ratio,
                                                                                   default_frame_range,
                                                                                   datasets_sampling_parameters,
                                                                                   datasets_sampling_weight,
                                                                                   data_processor)

    def get(self, index: int, rng_engine: np.random.Generator):
        index_of_dataset, index_of_sequence = self.sequence_picker[index]
        self.current_index_of_dataset = index_of_dataset
        self.current_index_of_sequence = index_of_sequence
        self._update_sample_type(rng_engine)
        data = self.do_sampling(rng_engine)
        assert data is not None
        return data

    def __len__(self):
        return len(self.sequence_picker)
