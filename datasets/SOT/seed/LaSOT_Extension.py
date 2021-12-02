from datasets.base.factory_seed import BaseSeed
from datasets.types.data_split import DataSplit


class LaSOT_Extension_Seed(BaseSeed):
    def __init__(self, root_path: str=None):
        if root_path is None:
            root_path = self.get_path_from_config('LaSOT_Extension_PATH')
        super(LaSOT_Extension_Seed, self).__init__('LaSOT_Extension', root_path, DataSplit.Full, 1)

    def construct(self, constructor):
        from .Impl.LaSOT_Extension import construct_LaSOT_Extension
        construct_LaSOT_Extension(constructor, self)
