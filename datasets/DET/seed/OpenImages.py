from datasets.base.factory_seed import BaseSeed
from datasets.types.data_split import DataSplit


class OpenImages_Seed(BaseSeed):
    def __init__(self, root_path: str=None, data_split=DataSplit.Training | DataSplit.Validation | DataSplit.Testing):
        if root_path is None:
            root_path = self.get_path_from_config('Open_Images_PATH')
        super(OpenImages_Seed, self).__init__('Open-Images-V6', root_path, data_split, 2)

    def construct(self, constructor):
        from .impl.OpenImages import construct_OpenImages
        construct_OpenImages(constructor, self)
