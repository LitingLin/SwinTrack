from datasets.base.common.manipulator import _BaseDatasetManipulator, DatasetObjectManipulatorIterator


class ImageDatasetImageManipulator:
    def __init__(self, dataset: dict, index_of_image: int, parent_iterator=None):
        self.image = dataset['images'][index_of_image]
        self.dataset = dataset
        self.index_of_image = index_of_image
        self.parent_iterator = parent_iterator

    def get_image_size(self):
        return self.image['size']

    def set_name(self, name: str):
        self.image['name'] = name

    def __len__(self):
        if 'objects' not in self.image:
            return 0
        return len(self.image['objects'])

    def __iter__(self):
        return DatasetObjectManipulatorIterator(self.image)

    def delete(self):
        del self.dataset['images'][self.index_of_image]
        del self.image
        if self.parent_iterator is not None:
            self.parent_iterator.deleted()


class ImageDatasetImageManipulatorIterator:
    def __init__(self, dataset: dict):
        self.dataset = dataset
        self.index = 0

    def __next__(self):
        if self.index >= len(self.dataset['images']):
            raise StopIteration
        modifier = ImageDatasetImageManipulator(self.dataset, self.index, self)
        self.index += 1
        return modifier

    def deleted(self):
        self.index -= 1


class ImageDatasetManipulator(_BaseDatasetManipulator):
    def apply_index_filter(self, indices: list):
        self.dataset['images'] = [self.dataset['images'][index] for index in indices]

    def __len__(self):
        return len(self.dataset['images'])

    def __iter__(self):
        return ImageDatasetImageManipulatorIterator(self.dataset)
