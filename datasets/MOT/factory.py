from datasets.base.factory import DatasetFactory
from datasets.base.video.dataset import VideoDataset
from datasets.types.specialized_dataset import SpecializedVideoDatasetType
from datasets.base.video.filter.func import apply_filters_on_video_dataset_
from datasets.MOT.dataset import MultipleObjectTrackingDataset_MemoryMapped
from typing import List

__all__ = ['MultipleObjectTrackingDatasetFactory']


class MultipleObjectTrackingDatasetFactory(DatasetFactory):
    def __init__(self, seeds: list):
        super(MultipleObjectTrackingDatasetFactory, self).__init__(seeds, VideoDataset,
                                                                   SpecializedVideoDatasetType.MultipleObjectTracking,
                                                                   apply_filters_on_video_dataset_,
                                                                   SpecializedVideoDatasetType.MultipleObjectTracking,
                                                                   MultipleObjectTrackingDataset_MemoryMapped)

    def construct(self, filters: list=None, cache_base_format: bool=True, dump_human_readable: bool=False) -> List[MultipleObjectTrackingDataset_MemoryMapped]:
        return super(MultipleObjectTrackingDatasetFactory, self).construct(filters, cache_base_format, dump_human_readable)

    def construct_as_base_interface(self, filters=None, make_cache=False, dump_human_readable=False) -> List[VideoDataset]:
        return super(MultipleObjectTrackingDatasetFactory, self).construct_as_base_interface(filters, make_cache, dump_human_readable)
