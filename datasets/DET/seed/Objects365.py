from datasets.base.factory_seed import BaseSeed
from datasets.types.data_split import DataSplit


class Objects365_Seed(BaseSeed):
    def __init__(self, root_path=None, data_split=DataSplit.Training | DataSplit.Validation):
        if root_path is None:
            root_path = self.get_path_from_config('Objects365_PATH')
        super(BaseSeed, self).__init__('Objects365', root_path, data_split, 1)

    def construct(self, constructor):
        from .impl.Objects365 import construct_Objects365
        construct_Objects365(constructor, self)
