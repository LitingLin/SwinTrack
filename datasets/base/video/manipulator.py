from datasets.base.common.manipulator import _BaseDatasetManipulator, DatasetObjectManipulatorIterator


class VideoDatasetFrameManipulator:
    def __init__(self, sequence: dict, index_of_frame: int, parent_iterator=None):
        self.frame = sequence['frames'][index_of_frame]
        self.sequence = sequence
        self.index_of_frame = index_of_frame
        self.parent_iterator = parent_iterator

    def get_image_size(self):
        return self.frame['size']

    def __len__(self):
        if 'objects' not in self.frame:
            return 0
        return len(self.frame['objects'])

    def __iter__(self):
        return DatasetObjectManipulatorIterator(self.frame)

    def delete(self):
        del self.sequence['frames'][self.index_of_frame]
        del self.frame
        if self.parent_iterator is not None:
            self.parent_iterator.deleted()


class VideoDatasetFrameManipulatorIterator:
    def __init__(self, sequence: dict):
        self.sequence = sequence
        self.index = 0

    def __next__(self):
        if self.index >= len(self.sequence['frames']):
            raise StopIteration

        modifier = VideoDatasetFrameManipulator(self.sequence, self.index, self)
        self.index += 1
        return modifier

    def deleted(self):
        self.index -= 1


class VideoDatasetFrameManipulatorReverseIterator:
    def __init__(self, sequence: dict):
        self.sequence = sequence

    def __iter__(self):
        self.index = len(self.sequence['frames']) - 1
        return self

    def __next__(self):
        if self.index < 0 or self.index >= len(self.sequence['frames']):
            raise StopIteration

        modifier = VideoDatasetFrameManipulator(self.sequence, self.index, self)
        self.index -= 1
        return modifier

    def deleted(self):
        pass


class VideoDatasetSequenceObjectManipulator:
    def __init__(self, sequence: dict, index_of_object, parent_iterator=None):
        self.object_ = sequence['objects'][index_of_object]
        self.sequence = sequence
        self.index_of_object = index_of_object
        self.parent_iterator = parent_iterator

    def get_id(self):
        return self.object_['id']

    def get_category_id(self):
        return self.object_['category_id']

    def has_category_id(self):
        return 'category_id' in self.object_

    def delete(self):
        del self.sequence['objects'][self.index_of_object]
        del self.object_
        if self.parent_iterator is not None:
            self.parent_iterator.deleted()


class VideoDatasetSequenceObjectManipulatorIterator:
    def __init__(self, sequence: dict):
        self.sequence = sequence

    def __iter__(self):
        self.index = 0
        return self

    def __next__(self):
        if self.index >= len(self.sequence['objects']):
            raise StopIteration

        modifier = VideoDatasetSequenceObjectManipulator(self.sequence, self.index, self)
        self.index += 1
        return modifier

    def deleted(self):
        self.index -= 1


class VideoDatasetSequenceManipulator:
    def __init__(self, dataset: dict, index_of_sequence: int, parent_iterator=None):
        self.sequence = dataset['sequences'][index_of_sequence]
        self.dataset = dataset
        self.index_of_sequence = index_of_sequence
        self.parent_iterator = parent_iterator

    def set_name(self, name: str):
        self.sequence['name'] = name

    def __len__(self):
        return len(self.sequence['frames'])

    def get_object_iterator(self):
        if 'objects' not in self.sequence:
            return ()
        return VideoDatasetSequenceObjectManipulatorIterator(self.sequence)

    def __iter__(self):
        return VideoDatasetFrameManipulatorIterator(self.sequence)

    def get_reverse_iterator(self):
        return VideoDatasetFrameManipulatorReverseIterator(self.sequence)

    def get_sequence_frame_size(self, allow_estimation = True):
        assert len(self.sequence['frames']) != 0
        from miscellanies.most_frequent import get_most_frequent_items_from_list
        sizes = []
        for frame in self.sequence['frames']:
            sizes.append(frame['size'])
        sizes = get_most_frequent_items_from_list(sizes, 2)
        if not allow_estimation:
            assert len(sizes) == 1
        return sizes[0][0]

    def delete(self):
        del self.dataset['sequences'][self.index_of_sequence]
        del self.sequence
        if self.parent_iterator is not None:
            self.parent_iterator.deleted()


class VideoDatasetSequenceManipulatorIterator:
    def __init__(self, dataset: dict):
        self.dataset = dataset
        self.index = 0

    def __next__(self):
        if self.index >= len(self.dataset['sequences']):
            raise StopIteration
        modifier = VideoDatasetSequenceManipulator(self.dataset, self.index, self)
        self.index += 1
        return modifier

    def deleted(self):
        self.index -= 1


class VideoDatasetManipulator(_BaseDatasetManipulator):
    def apply_index_filter(self, indices: list):
        self.dataset['sequences'] = [self.dataset['sequences'][index] for index in indices]

    def __len__(self):
        return len(self.dataset['sequences'])

    def __iter__(self):
        return VideoDatasetSequenceManipulatorIterator(self.dataset)
