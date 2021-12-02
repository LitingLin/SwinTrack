from datasets.base.factory_seed import BaseSeed
from datasets.types.data_split import DataSplit


class GOT10k_Seed(BaseSeed):
    def __init__(self, root_path: str=None, data_split: DataSplit = DataSplit.Training | DataSplit.Validation, sequence_filter=None):
        if root_path is None:
            root_path = self.get_path_from_config('GOT10k_PATH')
        self.sequence_filter = sequence_filter
        name = 'GOT-10k'
        if sequence_filter is not None:
            name += '-'
            name += sequence_filter
        super(GOT10k_Seed, self).__init__(name, root_path, data_split, 2)

    def construct(self, constructor):
        from .Impl.GOT10k import construct_GOT10k
        construct_GOT10k(constructor, self)
