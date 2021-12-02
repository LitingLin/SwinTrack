from datasets.base.storage_engine.memory_mapped import ListMemoryMapped
from miscellanies.platform_style_path import join_path
from datasets.base.common.memory_mapped.dataset import LazyAttributesLoader, DummyAttributesLoader, MemoryMappedDataset


__version__ = 1
__all__ = ['ImageClassificationDataset_MemoryMapped']


class ImageClassificationDatasetImage_MemoryMapped:
    def __init__(self, root_path: str, image_attributes: dict,
                 image_additional_attributes_loader: LazyAttributesLoader):
        self.root_path = root_path
        self.image_attributes = image_attributes
        self.image_additional_attributes = image_additional_attributes_loader

    def get_image_path(self):
        return join_path(self.root_path, self.image_attributes['path'])

    def get_image_size(self):
        return self.image_attributes['size']

    def get_category_id(self):
        return self.image_attributes['category_id']

    def has_attribute(self, name: str):
        return self.image_additional_attributes.has_attribute(name)

    def get_attribute(self, name: str):
        return self.image_additional_attributes.get_attribute(name)

    def get_all_attribute_name(self):
        return self.image_additional_attributes.get_all_attribute_name()


class ImageClassificationDataset_MemoryMapped(MemoryMappedDataset):
    def __init__(self, root_path: str, storage: ListMemoryMapped):
        super(ImageClassificationDataset_MemoryMapped, self).__init__(root_path, storage, __version__, 'ImageClassification')

    @staticmethod
    def load(path: str, root_path: str):
        return ImageClassificationDataset_MemoryMapped(root_path, MemoryMappedDataset.load_storage(path))

    def __getitem__(self, index: int):
        image_attribute = self.storage[self.index_matrix[index, 0]]

        image_additional_attributes_index = self.index_matrix[index, 1]

        if image_additional_attributes_index != -1:
            image_additional_attributes_loader = LazyAttributesLoader(self.storage, image_additional_attributes_index)
        else:
            image_additional_attributes_loader = DummyAttributesLoader()

        return ImageClassificationDatasetImage_MemoryMapped(self.root_path, image_attribute, image_additional_attributes_loader)
