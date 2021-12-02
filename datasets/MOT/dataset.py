from datasets.base.storage_engine.memory_mapped import ListMemoryMapped
import numpy as np
from miscellanies.platform_style_path import join_path
from datasets.base.common.memory_mapped.dataset import LazyAttributesLoader, DummyAttributesLoader, MemoryMappedDataset

__all__ = ['MultipleObjectTrackingDataset_MemoryMapped']
__version__ = 1


class MultipleObjectTrackingDatasetFrameObject_MemoryMapped:
    def __init__(self, frame_index: int, object_id: int, object_attribute, bounding_box: np.ndarray,
                 bounding_box_validity_flag: np.ndarray,
                 sequence_additional_attributes: LazyAttributesLoader):
        self.frame_index = frame_index
        self.object_id = object_id
        self.object_attribute = object_attribute
        self.bounding_box = bounding_box
        self.bounding_box_validity_flag = bounding_box_validity_flag
        self.sequence_additional_attributes = sequence_additional_attributes

    def get_id(self):
        return self.object_id

    def has_category_id(self):
        return 'category_id' in self.object_attribute

    def get_category_id(self):
        return self.object_attribute['category_id']

    def get_frame_index(self):
        return self.frame_index

    def has_bounding_box(self):
        return self.bounding_box is not None

    def get_bounding_box(self):
        return self.bounding_box

    def has_bounding_box_validity_flag(self):
        return self.bounding_box_validity_flag is not None

    def get_bounding_box_validity_flag(self):
        return self.bounding_box_validity_flag

    def get_attribute(self, name: str):
        return self.sequence_additional_attributes.get_attribute_tree_query(
            ('frames', self.frame_index, 'objects', self.object_id, name))

    def has_attribute(self, name: str):
        return self.sequence_additional_attributes.has_attribute_tree_query(
            ('frames', self.frame_index, 'objects', self.object_id, name))

    def get_all_attribute_name(self):
        return self.sequence_additional_attributes.get_all_attribute_name_tree_query(
            ('frames', self.frame_index, 'objects', self.object_id))


class MultipleObjectTrackingDatasetFrame_MemoryMapped:
    def __init__(self, root_path: str, sequence_attributes: dict, index_of_frame: int, frame_attributes: dict,
                 image_size,
                 frame_object_id_vector: np.ndarray,
                 frame_object_bounding_box_matrix: np.ndarray,
                 frame_object_bounding_box_validity_flag_vector: np.ndarray,
                 sequence_additional_attributes: LazyAttributesLoader):
        self.root_path = root_path
        self.sequence_attributes = sequence_attributes
        self.index_of_frame = index_of_frame
        self.frame_attributes = frame_attributes
        self.image_size = image_size
        self.frame_object_id_vector = frame_object_id_vector
        self.frame_object_bounding_box_matrix = frame_object_bounding_box_matrix
        self.frame_object_bounding_box_validity_flag_vector = frame_object_bounding_box_validity_flag_vector
        self.sequence_additional_attributes = sequence_additional_attributes

    def get_image_path(self):
        return join_path(self.root_path, self.sequence_attributes['path'], self.frame_attributes['path'])

    def get_image_size(self):
        return self.image_size

    def get_all_object_id(self):
        return self.frame_object_id_vector

    def has_bounding_box(self):
        return self.frame_object_bounding_box_matrix is not None

    def has_bounding_box_validity_flag(self):
        return self.frame_object_bounding_box_validity_flag_vector is not None

    def get_all_bounding_box_validity_flag(self):
        return self.frame_object_bounding_box_validity_flag_vector

    def get_all_bounding_box(self):
        return self.frame_object_bounding_box_matrix

    def get_attribute(self, name: str):
        return self.sequence_additional_attributes.get_attribute_tree_query(('frames', self.index_of_frame, name))

    def has_attribute(self, name: str):
        return self.sequence_additional_attributes.has_attribute_tree_query(('frames', self.index_of_frame, name))

    def get_all_attribute_name(self):
        return self.sequence_additional_attributes.get_all_attribute_name_tree_query(('frames', self.index_of_frame))

    def has_object(self, id_: int):
        return id_ in self.frame_object_id_vector

    def get_object_by_id(self, id_: int):
        index = np.where(self.frame_object_id_vector == id_)[0].item()
        return self[index]

    def __getitem__(self, index: int):
        object_id = self.frame_object_id_vector[index].item()
        object_attribute = self.sequence_attributes['objects'][object_id]

        frame_object_bounding_box = None
        frame_object_bounding_box_validity_flag = None

        if self.frame_object_bounding_box_matrix is not None:
            frame_object_bounding_box = self.frame_object_bounding_box_matrix[index, :]
        if self.frame_object_bounding_box_validity_flag_vector is not None:
            frame_object_bounding_box_validity_flag = self.frame_object_bounding_box_validity_flag_vector[index]

        return MultipleObjectTrackingDatasetFrameObject_MemoryMapped(self.index_of_frame,
                                                                     object_id,
                                                                     object_attribute,
                                                                     frame_object_bounding_box,
                                                                     frame_object_bounding_box_validity_flag,
                                                                     self.sequence_additional_attributes)

    def __len__(self):
        return self.frame_object_id_vector.shape[0]


class MultipleObjectTrackingDatasetSequenceObject_MemoryMapped:
    def __init__(self, object_id, object_attribute, object_frame_index_vector: np.ndarray,
                 object_bounding_box_matrix: np.ndarray,
                 object_bounding_box_validity_flag_vector: np.ndarray,
                 sequence_additional_attributes: LazyAttributesLoader):
        self.object_id = object_id
        self.object_attribute = object_attribute
        self.object_frame_index_vector = object_frame_index_vector
        self.object_bounding_box_matrix = object_bounding_box_matrix
        self.object_bounding_box_validity_flag_vector = object_bounding_box_validity_flag_vector
        self.sequence_additional_attributes = sequence_additional_attributes

    def has_category_id(self):
        return 'category_id' in self.object_attribute

    def get_category_id(self):
        return self.object_attribute['category_id']

    def get_id(self):
        return self.object_id

    def get_all_frame_index(self):
        return self.object_frame_index_vector

    def get_all_bounding_box_validity_flag(self):
        return self.object_bounding_box_validity_flag_vector

    def get_all_bounding_box(self):
        return self.object_bounding_box_matrix

    def get_attribute(self, name: str):
        return self.sequence_additional_attributes.get_attribute_tree_query(('objects', self.object_id, name))

    def has_attribute(self, name: str):
        return self.sequence_additional_attributes.has_attribute_tree_query(('objects', self.object_id, name))

    def get_all_attribute_name(self):
        return self.sequence_additional_attributes.get_all_attribute_name_tree_query(('objects', self.object_id))

    def __getitem__(self, index: int):
        frame_object_bounding_box = None
        frame_object_bounding_box_validity_flag = None

        if self.object_bounding_box_matrix is not None:
            frame_object_bounding_box = self.object_bounding_box_matrix[index]
        if self.object_bounding_box_validity_flag_vector is not None:
            frame_object_bounding_box_validity_flag = self.object_bounding_box_validity_flag_vector[index]
        return MultipleObjectTrackingDatasetFrameObject_MemoryMapped(self.object_frame_index_vector[index],
                                                                     self.object_id,
                                                                     self.object_attribute,
                                                                     frame_object_bounding_box,
                                                                     frame_object_bounding_box_validity_flag,
                                                                     self.sequence_additional_attributes)

    def __len__(self):
        return self.object_frame_index_vector.shape[0]


class MultipleObjectTrackingDatasetSequence_MemoryMapped_ObjectIterator:
    def __init__(self, sequence):
        self.sequence = sequence

    def __getitem__(self, index):
        return self.sequence.get_object(index)

    def __len__(self):
        return self.sequence.get_number_of_objects()


class MultipleObjectTrackingDatasetSequence_MemoryMapped_FrameIterator:
    def __init__(self, sequence):
        self.sequence = sequence

    def __getitem__(self, index):
        return self.sequence.get_frame(index)

    def __len__(self):
        return self.sequence.get_number_of_frames()


class MultipleObjectTrackingDatasetSequence_MemoryMapped:
    def __init__(self, root_path: str, sequence_attributes: dict, sequence_image_size_matrix: np.ndarray,
                 sequence_frame_object_attribute_indices_vector: np.ndarray,
                 sequence_frame_object_id_vector: np.ndarray,
                 sequence_object_frame_index_vector: np.ndarray,
                 sequence_object_bounding_box_matrix: np.ndarray,
                 sequence_object_bounding_box_validity_flag_matrix: np.ndarray,
                 sequence_additional_attributes: LazyAttributesLoader):
        self.root_path = root_path
        self.sequence_attributes = sequence_attributes
        self.object_ids = list(sequence_attributes['objects'].keys())
        self.sequence_image_size_matrix = sequence_image_size_matrix
        self.sequence_frame_object_attribute_indices_vector = sequence_frame_object_attribute_indices_vector
        self.sequence_frame_object_id_vector = sequence_frame_object_id_vector
        self.sequence_object_frame_index_vector = sequence_object_frame_index_vector
        self.sequence_object_bounding_box_matrix = sequence_object_bounding_box_matrix
        self.sequence_object_bounding_box_validity_flag_matrix = sequence_object_bounding_box_validity_flag_matrix
        self.sequence_additional_attributes = sequence_additional_attributes
        '''

        :param root_path:
        :param sequence_attributes:
        {
            'path': sequence_path
            'objects': [{'index_ranges: (begin, end)', 'category_id': category_id, 'id': id}, ...]
            'frames': [{'path': '', object_indices: [1,2,3]}]
        }
        :param object_indices:
        :param bounding_box_matrix:
        :param bounding_box_validity_vector:



        '''

    def has_fps(self):
        return 'fps' in self.sequence_attributes

    def get_fps(self):
        return self.sequence_attributes['fps']

    def get_object_iterator(self):
        return MultipleObjectTrackingDatasetSequence_MemoryMapped_ObjectIterator(self)

    def get_frame_iterator(self):
        return MultipleObjectTrackingDatasetSequence_MemoryMapped_FrameIterator(self)

    def get_name(self):
        return self.sequence_attributes['name']

    def get_number_of_objects(self):
        return len(self.sequence_attributes['objects'])

    def get_number_of_frames(self):
        return len(self.sequence_attributes['frames'])

    def get_frame(self, index: int):
        frame_attribute = self.sequence_attributes['frames'][index]
        object_attributes_index_range = frame_attribute['object_attributes_index_range']
        if self.sequence_image_size_matrix is None:
            image_size = self.sequence_attributes['frame_size']
        else:
            image_size = self.sequence_image_size_matrix[index, :]
        object_attribute_indices = self.sequence_frame_object_attribute_indices_vector[
                                   object_attributes_index_range[0]: object_attributes_index_range[1]]
        frame_object_id_vector = self.sequence_frame_object_id_vector[
                                 object_attributes_index_range[0]: object_attributes_index_range[1]]
        if self.sequence_object_bounding_box_matrix is not None:
            frame_object_bounding_box_matrix = self.sequence_object_bounding_box_matrix[object_attribute_indices]
        else:
            frame_object_bounding_box_matrix = None
        if self.sequence_object_bounding_box_validity_flag_matrix is not None:
            frame_object_bounding_box_validity_flag_matrix = self.sequence_object_bounding_box_validity_flag_matrix[
                object_attribute_indices]
        else:
            frame_object_bounding_box_validity_flag_matrix = None
        return MultipleObjectTrackingDatasetFrame_MemoryMapped(self.root_path, self.sequence_attributes, index,
                                                               frame_attribute, image_size, frame_object_id_vector,
                                                               frame_object_bounding_box_matrix,
                                                               frame_object_bounding_box_validity_flag_matrix,
                                                               self.sequence_additional_attributes)

    def get_object(self, index: int):
        object_id = self.object_ids[index]
        object_attribute = self.sequence_attributes['objects'][object_id]
        object_attributes_index_range = object_attribute['object_attributes_index_range']
        object_frame_index_vector = self.sequence_object_frame_index_vector[
                                    object_attributes_index_range[0]: object_attributes_index_range[1]]
        if self.sequence_object_bounding_box_matrix is not None:
            object_bounding_box_matrix = self.sequence_object_bounding_box_matrix[
                                         object_attributes_index_range[0]: object_attributes_index_range[1], :]
        else:
            object_bounding_box_matrix = None
        if self.sequence_object_bounding_box_validity_flag_matrix is not None:
            object_bounding_box_validity_flag_vector = self.sequence_object_bounding_box_validity_flag_matrix[
                                                       object_attributes_index_range[0]: object_attributes_index_range[
                                                           1]]
        else:
            object_bounding_box_validity_flag_vector = None

        return MultipleObjectTrackingDatasetSequenceObject_MemoryMapped(object_id, object_attribute,
                                                                        object_frame_index_vector,
                                                                        object_bounding_box_matrix,
                                                                        object_bounding_box_validity_flag_vector,
                                                                        self.sequence_additional_attributes)

    def get_attribute(self, name: str):
        return self.sequence_additional_attributes.get_attribute(name)

    def has_attribute(self, name: str):
        return self.sequence_additional_attributes.has_attribute(name)

    def get_all_attribute_name(self):
        return self.sequence_additional_attributes.get_all_attribute_name()

    def __getitem__(self, index: int):
        return self.get_frame(index)

    def __len__(self):
        return self.get_number_of_frames()

    def has_bounding_box(self):
        return self.sequence_object_bounding_box_matrix is not None

    def has_bounding_box_validity_flag(self):
        return self.sequence_object_bounding_box_validity_flag_matrix is not None


class MultipleObjectTrackingDataset_MemoryMapped(MemoryMappedDataset):
    def __init__(self, root_path: str, storage: ListMemoryMapped):
        super(MultipleObjectTrackingDataset_MemoryMapped, self).__init__(root_path, storage, __version__,
                                                                         'MultipleObjectTracking')

    @staticmethod
    def load(path: str, root_path: str):
        return MultipleObjectTrackingDataset_MemoryMapped(root_path, MemoryMappedDataset.load_storage(path))

    def __getitem__(self, index: int):
        sequence_attributes = self.storage[self.index_matrix[index, 0]]

        sequence_image_size_matrix_index = self.index_matrix[index, 1]
        sequence_image_size_matrix = self.storage[
            sequence_image_size_matrix_index] if sequence_image_size_matrix_index != -1 else None
        sequence_frame_object_attribute_indices_vector_index = self.index_matrix[index, 2]
        sequence_frame_object_attribute_indices_vector = self.storage[
            sequence_frame_object_attribute_indices_vector_index] if sequence_frame_object_attribute_indices_vector_index != -1 else None
        sequence_frame_object_id_vector_index = self.index_matrix[index, 3]
        sequence_frame_object_id_vector = self.storage[
            sequence_frame_object_id_vector_index] if sequence_frame_object_id_vector_index != -1 else None
        sequence_object_frame_index_vector_index = self.index_matrix[index, 4]
        sequence_object_frame_index_vector = self.storage[
            sequence_object_frame_index_vector_index] if sequence_object_frame_index_vector_index != -1 else None
        sequence_object_bounding_box_matrix_index = self.index_matrix[index, 5]
        sequence_object_bounding_box_matrix = self.storage[
            sequence_object_bounding_box_matrix_index] if sequence_object_bounding_box_matrix_index != -1 else None
        sequence_object_bounding_box_validity_flag_matrix_index = self.index_matrix[index, 6]
        sequence_object_bounding_box_validity_flag_matrix = self.storage[
            sequence_object_bounding_box_validity_flag_matrix_index] if sequence_object_bounding_box_validity_flag_matrix_index != -1 else None
        sequence_additional_attributes_index = self.index_matrix[index, 7]
        if sequence_additional_attributes_index != -1:
            sequence_additional_attributes = LazyAttributesLoader(self.storage, sequence_additional_attributes_index)
        else:
            sequence_additional_attributes = DummyAttributesLoader()

        return MultipleObjectTrackingDatasetSequence_MemoryMapped(self.root_path,
                                                                  sequence_attributes, sequence_image_size_matrix,
                                                                  sequence_frame_object_attribute_indices_vector,
                                                                  sequence_frame_object_id_vector,
                                                                  sequence_object_frame_index_vector,
                                                                  sequence_object_bounding_box_matrix,
                                                                  sequence_object_bounding_box_validity_flag_matrix,
                                                                  sequence_additional_attributes)
