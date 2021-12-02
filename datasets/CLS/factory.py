from datasets.base.factory import DatasetFactory
from datasets.base.image.dataset import ImageDataset
from datasets.types.specialized_dataset import SpecializedImageDatasetType
from datasets.base.image.filter.func import apply_filters_on_image_dataset_
from datasets.CLS.dataset import ImageClassificationDataset_MemoryMapped
from typing import List

__all__ = ['ImageClassificationDatasetFactory']


class ImageClassificationDatasetFactory(DatasetFactory):
    def __init__(self, seeds: list):
        super(ImageClassificationDatasetFactory, self).__init__(seeds, ImageDataset,
                                                                SpecializedImageDatasetType.Classification,
                                                                apply_filters_on_image_dataset_,
                                                                SpecializedImageDatasetType.Classification,
                                                                ImageClassificationDataset_MemoryMapped)

    def construct(self, filters: list = None, cache_base_format: bool = True, dump_human_readable: bool = False) -> List[ImageClassificationDataset_MemoryMapped]:
        return super(ImageClassificationDatasetFactory, self).construct(filters, cache_base_format, dump_human_readable)

    def construct_as_base_interface(self, filters=None, make_cache=False, dump_human_readable=False) -> List[ImageDataset]:
        return super(ImageClassificationDatasetFactory, self).construct_as_base_interface(filters, make_cache, dump_human_readable)
