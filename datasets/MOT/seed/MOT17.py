from datasets.types.data_split import DataSplit
from datasets.base.factory_seed import BaseSeed


class MOT17_Seed(BaseSeed):
    def __init__(self, root_path: str=None):
        if root_path is None:
            root_path = self.get_path_from_config('MOT17_PATH')
        super(MOT17_Seed, self).__init__('MOT17', root_path, DataSplit.Full, 1)

    def construct(self, constructor):
        from .Impl.MOT17 import construct_MOT17
        construct_MOT17(constructor, self)
