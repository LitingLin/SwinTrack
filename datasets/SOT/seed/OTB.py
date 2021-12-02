from datasets.base.factory_seed import BaseSeed
from datasets.types.data_split import DataSplit
from enum import Enum, auto


class OTBVersion(Enum):
    OTB100 = auto()
    OTB50 = auto()
    OTB2013 = auto()
    OTB2015 = auto()


class OTB_Seed(BaseSeed):
    def __init__(self, root_path: str=None, version: OTBVersion=OTBVersion.OTB100):
        if root_path is None:
            root_path = self.get_path_from_config('OTB_PATH')

        self.otb_version = version

        super(OTB_Seed, self).__init__(version.name, root_path, DataSplit.Full, 4)

    def construct(self, constructor):
        from .Impl.OTB import construct_OTB
        construct_OTB(constructor, self)
