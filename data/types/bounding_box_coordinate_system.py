import enum


class BoundingBoxCoordinateSystem(enum.Enum):
    Rasterized = enum.auto() # index of pixels
    Spatial = enum.auto()
