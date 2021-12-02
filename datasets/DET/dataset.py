from datasets.base.storage_engine.memory_mapped import ListMemoryMapped
import numpy as np
from miscellanies.platform_style_path import join_path
from datasets.base.common.memory_mapped.dataset import LazyAttributesLoader, DummyAttributesLoader, MemoryMappedDataset

__version__ = 1
__all__ = ['DetectionDataset_MemoryMapped']


class DetectionDatasetObject_MemoryMapped:
    def __init__(self, object_attributes: dict, object_index: int, bounding_box: np.ndarray,
                 bounding_box_validity_flag: np.ndarray, image_additional_attributes: LazyAttributesLoader):
        self.object_attributes = object_attributes
        self.object_index = object_index
        self.bounding_box = bounding_box
        self.bounding_box_validity_flag = bounding_box_validity_flag
        self.image_additional_attributes = image_additional_attributes

    def has_bounding_box(self):
        return self.bounding_box is not None

    def has_bounding_box_validity_flag(self):
        return self.bounding_box_validity_flag is not None

    def has_category_id(self):
        return 'category_id' in self.object_attributes

    def get_bounding_box(self):
        return self.bounding_box

    def get_bounding_box_validity_flag(self):
        return self.bounding_box_validity_flag

    def get_category_id(self):
        return self.object_attributes['category_id']

    def get_attribute(self, name: str):
        return self.image_additional_attributes.get_attribute_tree_query(('objects', self.object_index, name))

    def has_attribute(self, name: str):
        return self.image_additional_attributes.has_attribute_tree_query(('objects', self.object_index, name))

    def get_all_attribute_name(self):
        return self.image_additional_attributes.get_all_attribute_name_tree_query(('objects', self.object_index))


class DetectionDatasetImage_MemoryMapped:
    def __init__(self, root_path: str, image_attributes: dict, bounding_box_matrix: np.ndarray,
                 bounding_box_validity_flag_matrix: np.ndarray,
                 image_additional_attributes_loader: LazyAttributesLoader):
        self.root_path = root_path
        self.image_attributes = image_attributes
        self.bounding_box_matrix = bounding_box_matrix
        self.bounding_box_validity_flag_vector = bounding_box_validity_flag_matrix
        self.image_additional_attributes = image_additional_attributes_loader

    def get_image_path(self):
        return join_path(self.root_path, self.image_attributes['path'])

    def get_image_size(self):
        return self.image_attributes['size']

    def has_category_id(self):
        return 'category_id' in self.image_attributes

    def get_category_id(self):
        return self.image_attributes['category_id']

    def __getitem__(self, index: int):
        if index >= len(self):
            raise IndexError
        bounding_box = self.bounding_box_matrix[index, :] if self.bounding_box_matrix is not None else None
        bounding_box_validity_flag = self.bounding_box_validity_flag_vector[index] if self.bounding_box_validity_flag_vector is not None else None

        return DetectionDatasetObject_MemoryMapped(self.image_attributes['objects'][index], index, bounding_box,
                                                   bounding_box_validity_flag, self.image_additional_attributes)

    def __len__(self):
        return len(self.image_attributes['objects'])

    def get_all_bounding_box(self):
        return self.bounding_box_matrix

    def get_all_bounding_box_validity_flag(self):
        return self.bounding_box_validity_flag_vector

    def has_bounding_box(self):
        return self.bounding_box_matrix is not None

    def has_bounding_box_validity_flag(self):
        return self.bounding_box_validity_flag_vector is not None

    def has_attribute(self, name: str):
        return self.image_additional_attributes.has_attribute(name)

    def get_attribute(self, name: str):
        return self.image_additional_attributes.get_attribute(name)

    def get_all_attribute_name(self):
        return self.image_additional_attributes.get_all_attribute_name()


class DetectionDataset_MemoryMapped(MemoryMappedDataset):
    def __init__(self, root_path: str, storage: ListMemoryMapped):
        super(DetectionDataset_MemoryMapped, self).__init__(root_path, storage, __version__, 'Detection')

    @staticmethod
    def load(path: str, root_path: str):
        return DetectionDataset_MemoryMapped(root_path, MemoryMappedDataset.load_storage(path))

    def __getitem__(self, index: int):
        image_attribute = self.storage[self.index_matrix[index, 0]]

        bounding_box_matrix_index = self.index_matrix[index, 1]
        bounding_box_matrix = self.storage[bounding_box_matrix_index] if bounding_box_matrix_index != -1 else None

        bounding_box_validity_flag_vector_index = self.index_matrix[index, 2]
        bounding_box_validity_flag_vector = self.storage[
            bounding_box_validity_flag_vector_index] if bounding_box_validity_flag_vector_index != -1 else None

        image_additional_attributes_index = self.index_matrix[index, 3]

        if image_additional_attributes_index != -1:
            image_additional_attributes_loader = LazyAttributesLoader(self.storage, image_additional_attributes_index)
        else:
            image_additional_attributes_loader = DummyAttributesLoader()

        return DetectionDatasetImage_MemoryMapped(self.root_path,
                                                  image_attribute, bounding_box_matrix,
                                                  bounding_box_validity_flag_vector,
                                                  image_additional_attributes_loader)
