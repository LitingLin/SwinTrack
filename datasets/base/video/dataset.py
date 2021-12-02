import os
from miscellanies.platform_style_path import join_path
from .manipulator import VideoDatasetManipulator
from datasets.types.specialized_dataset import SpecializedVideoDatasetType
from datasets.base.common.dataset import _BaseDataset, _BaseDatasetObject

__all__ = ['VideoDataset']
__version__ = 1


class VideoDatasetFrame:
    def __init__(self, frame: dict, root_path: str, sequence_path: str):
        self.frame = frame
        self.root_path = root_path
        self.sequence_path = sequence_path

    def get_image_file_name(self):
        return os.path.basename(self.frame['path'])

    def get_image_path(self):
        return join_path(self.root_path, self.sequence_path, self.frame['path'])

    def get_image_size(self):
        return self.frame['size']

    def has_attribute(self, name: str):
        return name in self.frame

    def get_attribute(self, name: str):
        return self.frame[name]

    def get_all_attribute_name(self):
        return self.frame.keys()

    def __len__(self):
        return self.frame['objects']

    def __getitem__(self, index: int):
        return _BaseDatasetObject(self.frame['objects'][index])


class VideoDatasetSequenceObject:
    def __init__(self, object_: dict):
        self.object_ = object_

    def get_id(self):
        return self.object_['id']

    def get_category_id(self):
        return self.object_['category_id']

    def has_category_id(self):
        return 'category_id' in self.object_

    def has_attribute(self, name: str):
        return name in self.object_

    def get_attribute(self, name: str):
        return self.object_[name]

    def get_all_attribute_name(self):
        return self.object_.keys()


class VideoDatasetSequenceObjectIterator:
    def __init__(self, objects: list):
        self.objects = objects

    def __iter__(self):
        self.iter = iter(self.objects)
        return self

    def __next__(self):
        return VideoDatasetSequenceObject(next(self.iter))


class VideoDatasetSequence:
    def __init__(self, sequence: dict, root_path: str):
        self.sequence = sequence
        self.root_path = root_path

    def get_name(self):
        return self.sequence['name']

    def has_attribute(self, name: str):
        return name in self.sequence

    def get_attribute(self, name: str):
        return self.sequence[name]

    def get_all_attribute_name(self):
        return self.sequence.keys()

    def get_number_of_objects(self):
        if 'objects' in self.sequence:
            return len(self.sequence['objects'])
        return 0

    def get_object(self, index: int):
        return VideoDatasetSequenceObject(self.sequence['objects'][index])

    def get_object_iterator(self):
        if self.get_number_of_objects() == 0:
            return ()
        return VideoDatasetSequenceObjectIterator(self.sequence['objects'])

    def __len__(self):
        return len(self.sequence['frames'])

    def __getitem__(self, index: int):
        frame = self.sequence['frames'][index]
        return VideoDatasetFrame(frame, self.root_path, self.sequence['path'])


class VideoDataset(_BaseDataset):
    def __init__(self, root_path: str, dataset: dict = None):
        super(VideoDataset, self).__init__(root_path, dataset)

    def get_root_path(self):
        return self.root_path

    def set_root_path(self, path: str):
        self.root_path = path

    def get_version(self):
        return self.dataset['version'][1]

    @staticmethod
    def load(yaml_path: str, root_path: str):
        return VideoDataset(root_path, _BaseDataset.load(yaml_path, __version__))

    def get_constructor(self, type_: SpecializedVideoDatasetType, version: int):
        assert isinstance(type_, SpecializedVideoDatasetType)
        if type_ == SpecializedVideoDatasetType.SingleObjectTracking:
            from datasets.SOT.constructor.base_interface import SingleObjectTrackingDatasetConstructorGenerator
            return SingleObjectTrackingDatasetConstructorGenerator(self.dataset, self.root_path, version)
        elif type_ == SpecializedVideoDatasetType.MultipleObjectTracking:
            from datasets.MOT.constructor.base_interface import MultipleObjectTrackingDatasetConstructorGenerator
            return MultipleObjectTrackingDatasetConstructorGenerator(self.dataset, self.root_path, version)
        else:
            raise NotImplementedError

    def specialize(self, type_: SpecializedVideoDatasetType, path: str):
        assert isinstance(type_, SpecializedVideoDatasetType)
        if type_ == SpecializedVideoDatasetType.SingleObjectTracking:
            from datasets.SOT.constructor.specialization import \
                construct_single_object_tracking_dataset_memory_mapped_from_base_video_dataset
            from datasets.SOT.dataset import SingleObjectTrackingDataset_MemoryMapped
            return SingleObjectTrackingDataset_MemoryMapped(self.root_path,
                                                            construct_single_object_tracking_dataset_memory_mapped_from_base_video_dataset(
                                                                self.dataset, path))
        elif type_ == SpecializedVideoDatasetType.MultipleObjectTracking:
            from datasets.MOT.constructor.specialization import \
                construct_multiple_object_tracking_dataset_memory_mapped_from_base_video_dataset
            from datasets.MOT.dataset import MultipleObjectTrackingDataset_MemoryMapped
            return MultipleObjectTrackingDataset_MemoryMapped(self.root_path,
                                                              construct_multiple_object_tracking_dataset_memory_mapped_from_base_video_dataset(
                                                                  self.dataset, path))
        else:
            raise NotImplementedError

    def get_manipulator(self):
        return VideoDatasetManipulator(self.dataset)

    def __getitem__(self, index: int):
        return VideoDatasetSequence(self.dataset['sequences'][index], self.root_path)

    def __len__(self):
        return len(self.dataset['sequences'])
