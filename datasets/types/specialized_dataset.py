from enum import Enum, auto


class SpecializedImageDatasetType(Enum):
    Classification = auto()
    Detection = auto()


class SpecializedVideoDatasetType(Enum):
    SingleObjectTracking = auto()
    MultipleObjectTracking = auto()
