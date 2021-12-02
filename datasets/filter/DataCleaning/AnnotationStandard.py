from datasets.filter._common import _BaseFilter
from data.types.bounding_box_format import BoundingBoxFormat
from data.types.pixel_coordinate_system import PixelCoordinateSystem
from data.types.bounding_box_coordinate_system import BoundingBoxCoordinateSystem
from data.types.pixel_definition import PixelDefinition


class DataCleaning_AnnotationStandard(_BaseFilter):
    def __init__(self, bounding_box_format: BoundingBoxFormat=None, pixel_coordinate_system: PixelCoordinateSystem=None,
                 bounding_box_coordinate_system: BoundingBoxCoordinateSystem=None,
                 pixel_definition: PixelDefinition=None):
        if isinstance(bounding_box_format, str):
            self.bounding_box_format = BoundingBoxFormat[bounding_box_format]
        else:
            self.bounding_box_format = bounding_box_format

        if isinstance(pixel_coordinate_system, str):
            self.pixel_coordinate_system = PixelCoordinateSystem[pixel_coordinate_system]
        else:
            self.pixel_coordinate_system = pixel_coordinate_system

        if isinstance(bounding_box_coordinate_system, str):
            self.bounding_box_coordinate_system = BoundingBoxCoordinateSystem[bounding_box_coordinate_system]
        else:
            self.bounding_box_coordinate_system = bounding_box_coordinate_system

        if isinstance(pixel_definition, str):
            self.pixel_definition = PixelDefinition[pixel_definition]
        else:
            self.pixel_definition = pixel_definition
