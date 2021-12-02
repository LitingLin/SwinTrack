from datasets.base.factory_seed import BaseSeed
from datasets.types.data_split import DataSplit
from typing import Iterable


class TrackingNet_Seed(BaseSeed):
    def __init__(self, root_path: str=None, data_split = DataSplit.Training, enable_set_ids: Iterable[int]=None, sequence_name_class_map_file_path: str=None):
        if root_path is None:
            root_path = self.get_path_from_config('TrackingNet_PATH')
        name = 'TrackingNet'
        if enable_set_ids is not None:
            name += '-'
            name += '_'.join([str(v) for v in enable_set_ids])
        super(TrackingNet_Seed, self).__init__(name, root_path, data_split, 3)
        self.sequence_name_class_map_file_path = sequence_name_class_map_file_path
        self.enable_set_ids = enable_set_ids

    def construct(self, constructor):
        from .Impl.TrackingNet import construct_TrackingNet
        construct_TrackingNet(constructor, self)
