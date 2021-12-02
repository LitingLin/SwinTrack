from datasets.base.factory_seed import BaseSeed
from datasets.types.data_split import DataSplit


class LaSOT_Seed(BaseSeed):
    def __init__(self, root_path: str=None, data_split: DataSplit = DataSplit.Training | DataSplit.Validation):
        if root_path is None:
            root_path = self.get_path_from_config('LaSOT_PATH')
        super(LaSOT_Seed, self).__init__('LaSOT', root_path, data_split, 2)

    def construct(self, constructor):
        from .Impl.LaSOT import construct_LaSOT
        construct_LaSOT(constructor, self)
