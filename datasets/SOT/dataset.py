from datasets.base.storage_engine.memory_mapped import ListMemoryMapped
import numpy as np
from miscellanies.platform_style_path import join_path
from datasets.base.common.memory_mapped.dataset import LazyAttributesLoader, DummyAttributesLoader, MemoryMappedDataset

__all__ = ['SingleObjectTrackingDataset_MemoryMapped']
__version__ = 1


class SingleObjectTrackingDatasetFrame_MemoryMapped:
    def __init__(self, root_path: str, sequence_attributes: dict,
                 frame_index: int, image_size,
                 bounding_box: np.ndarray, bounding_box_validity_flag: np.ndarray,
                 sequence_additional_attributes_loader: LazyAttributesLoader):
        self.root_path = root_path
        self.sequence_attributes = sequence_attributes
        self.frame_attributes = sequence_attributes['frames'][frame_index]
        self.frame_index = frame_index
        self.image_size = image_size
        self.bounding_box = bounding_box
        self.bounding_box_validity_flag = bounding_box_validity_flag
        self.sequence_additional_attributes_loader = sequence_additional_attributes_loader

    def get_bounding_box(self):
        return self.bounding_box

    def get_bounding_box_validity_flag(self):
        return self.bounding_box_validity_flag

    def get_image_size(self):
        return self.image_size

    def get_image_path(self):
        return join_path(self.root_path, self.sequence_attributes['path'], self.frame_attributes['path'])

    def has_bounding_box(self):
        return self.bounding_box is not None

    def has_bounding_box_validity_flag(self):
        return self.bounding_box_validity_flag is not None

    def get_all_frame_attribute_name(self):
        return self.sequence_additional_attributes_loader.get_all_attribute_name_tree_query(('frames', self.frame_index))

    def get_frame_attribute(self, name: str):
        return self.sequence_additional_attributes_loader.get_attribute_tree_query(('frames', self.frame_index, name))

    def has_frame_attribute(self, name: str):
        return self.sequence_additional_attributes_loader.has_attribute_tree_query(('frames', self.frame_index, name))

    def get_all_object_attribute_name(self):
        return self.sequence_additional_attributes_loader.get_all_attribute_name_tree_query(('frames', self.frame_index, 'object'))

    def get_object_attribute(self, name: str):
        return self.sequence_additional_attributes_loader.get_attribute_tree_query(('frames', self.frame_index, name, 'object'))

    def has_object_attribute(self, name: str):
        return self.sequence_additional_attributes_loader.has_attribute_tree_query(('frames', self.frame_index, name, 'object'))


class SingleObjectTrackingDatasetSequence_MemoryMapped:
    def __init__(self, root_path: str, sequence_attributes: dict,
                 image_size_matrix: np.ndarray,
                 bounding_box_matrix: np.ndarray, bounding_box_validity_flag_matrix: np.ndarray,
                 sequence_additional_attributes_loader:LazyAttributesLoader):
        self.root_path = root_path
        self.sequence_attributes = sequence_attributes
        self.image_size_matrix = image_size_matrix
        self.bounding_box_matrix = bounding_box_matrix
        self.bounding_box_validity_flag_vector = bounding_box_validity_flag_matrix
        self.sequence_additional_attributes = sequence_additional_attributes_loader

    def get_name(self):
        return self.sequence_attributes['name']

    def has_fps(self):
        return 'fps' in self.sequence_attributes

    def get_fps(self):
        return self.sequence_attributes['fps']

    def has_bounding_box(self):
        return self.bounding_box_matrix is not None

    def has_category_id(self):
        return 'category_id' in self.sequence_attributes

    def has_bounding_box_validity_flag(self):
        return self.bounding_box_validity_flag_vector is not None

    def has_sequence_attribute(self, name: str):
        return self.sequence_additional_attributes.has_attribute(name)

    def get_sequence_attribute(self, name: str):
        return self.sequence_additional_attributes.get_attribute(name)

    def get_all_sequence_attribute_name(self):
        return self.sequence_additional_attributes.get_all_attribute_name()

    def get_object_attribute(self, name: str):
        return self.sequence_additional_attributes.get_attribute_tree_query(('object', name))

    def has_object_attribute(self, name: str):
        return self.sequence_additional_attributes.has_attribute_tree_query(('object', name))

    def get_all_object_attribute_name(self):
        return self.sequence_additional_attributes.get_all_attribute_name_tree_query(('object',))

    def get_all_bounding_box(self):
        return self.bounding_box_matrix

    def get_all_bounding_box_validity_flag(self):
        return self.bounding_box_validity_flag_vector

    def get_category_id(self):
        return self.sequence_attributes['category_id']

    def __getitem__(self, index: int):
        if index >= len(self):
            raise IndexError
        if self.image_size_matrix is not None:
            image_size = self.image_size_matrix[index, :]
        else:
            image_size = self.sequence_attributes['frame_size']

        bounding_box = self.bounding_box_matrix[index, :] if self.bounding_box_matrix is not None else None

        bounding_box_validity_flag = self.bounding_box_validity_flag_vector[index] if self.bounding_box_validity_flag_vector is not None else None

        return SingleObjectTrackingDatasetFrame_MemoryMapped(self.root_path,
                                                             self.sequence_attributes, index, image_size,
                                                             bounding_box, bounding_box_validity_flag,
                                                             self.sequence_additional_attributes)

    def __len__(self):
        return len(self.sequence_attributes['frames'])


class SingleObjectTrackingDataset_MemoryMapped(MemoryMappedDataset):
    def __init__(self, root_path: str, storage: ListMemoryMapped):
        super(SingleObjectTrackingDataset_MemoryMapped, self).__init__(root_path, storage, __version__, 'SingleObjectTracking')

    @staticmethod
    def load(path: str, root_path: str):
        return SingleObjectTrackingDataset_MemoryMapped(root_path, MemoryMappedDataset.load_storage(path))

    def __getitem__(self, index: int):
        sequence_attribute = self.storage[self.index_matrix[index, 0]]

        image_size_matrix_index = self.index_matrix[index, 1]
        image_size_matrix = self.storage[image_size_matrix_index] if image_size_matrix_index != -1 else None

        bounding_box_matrix_index = self.index_matrix[index, 2]
        bounding_box_matrix = self.storage[bounding_box_matrix_index] if bounding_box_matrix_index != -1 else None

        bounding_box_validity_flag_vector_index = self.index_matrix[index, 3]
        bounding_box_validity_flag_vector = self.storage[
            bounding_box_validity_flag_vector_index] if bounding_box_validity_flag_vector_index != -1 else None

        sequence_additional_attributes_index = self.index_matrix[index, 4]

        if sequence_additional_attributes_index != -1:
            sequence_additional_attributes = LazyAttributesLoader(self.storage,
                                                                         sequence_additional_attributes_index)
        else:
            sequence_additional_attributes = DummyAttributesLoader()

        return SingleObjectTrackingDatasetSequence_MemoryMapped(self.root_path,
                                                                sequence_attribute, image_size_matrix,
                                                                bounding_box_matrix,
                                                                bounding_box_validity_flag_vector,
                                                                sequence_additional_attributes)
