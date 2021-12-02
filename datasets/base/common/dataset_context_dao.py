from data.types.pixel_coordinate_system import PixelCoordinateSystem
from data.types.bounding_box_coordinate_system import BoundingBoxCoordinateSystem
from data.types.bounding_box_format import BoundingBoxFormat
from data.types.pixel_definition import PixelDefinition


class DatasetContextDAO:
    def __init__(self, attributes: dict):
        self.attributes = attributes
        if 'context' in attributes:
            context = attributes['context']
            self.bounding_box_format = BoundingBoxFormat[context['bounding_box_format']]
            self.bounding_box_coordinate_system = BoundingBoxCoordinateSystem[context['bounding_box_coordinate_system']]
            self.pixel_definition = PixelDefinition[context['pixel_definition']]
            bounding_box_data_type = context['bounding_box_data_type']
            if bounding_box_data_type == 'float':
                bounding_box_data_type = float
            elif bounding_box_data_type == 'int':
                bounding_box_data_type = int
            else:
                raise RuntimeError(f'Unknown value {bounding_box_data_type}')
            self.bounding_box_data_type = bounding_box_data_type
            self.pixel_coordinate_system = PixelCoordinateSystem[context['pixel_coordinate_system']]

    def has_context(self):
        return 'context' in self.attributes

    def _try_allocate_context_object(self):
        if 'context' not in self.attributes:
            self.attributes['context'] = {}

    def set_bounding_box_format(self, bounding_box_format: BoundingBoxFormat):
        self._try_allocate_context_object()
        self.bounding_box_format = bounding_box_format
        self.attributes['context']['bounding_box_format'] = bounding_box_format.name

    def set_pixel_definition(self, pixel_definition: PixelDefinition):
        self._try_allocate_context_object()
        self.pixel_definition = pixel_definition
        self.attributes['context']['pixel_definition'] = pixel_definition.name

    def set_pixel_coordinate_system(self, pixel_coordinate_system: PixelCoordinateSystem):
        self._try_allocate_context_object()
        self.pixel_coordinate_system = pixel_coordinate_system
        self.attributes['context']['pixel_coordinate_system'] = pixel_coordinate_system.name

    def set_bounding_box_coordinate_system(self, bounding_box_coordinate_system: BoundingBoxCoordinateSystem):
        self._try_allocate_context_object()
        self.bounding_box_coordinate_system = bounding_box_coordinate_system
        self.attributes['context']['bounding_box_coordinate_system'] = bounding_box_coordinate_system.name

    def set_bounding_box_data_type(self, type_):
        self._try_allocate_context_object()
        if type_ == int:
            self.attributes['context']['bounding_box_data_type'] = 'int'
        elif type_ == float:
            self.attributes['context']['bounding_box_data_type'] = 'float'
        else:
            raise RuntimeError(f'Unknown value {type_}')
        self.bounding_box_data_type = type_

    def get_bounding_box_format(self):
        return self.bounding_box_format

    def get_pixel_definition(self):
        return self.pixel_definition

    def get_pixel_coordinate_system(self):
        return self.pixel_coordinate_system

    def get_bounding_box_coordinate_system(self):
        return self.bounding_box_coordinate_system

    def get_bounding_box_data_type(self):
        return self.bounding_box_data_type
