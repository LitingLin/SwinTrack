from .image import get_image_center_point
from data.types.pixel_coordinate_system import PixelCoordinateSystem
from data.types.pixel_definition import PixelDefinition


def bbox_horizontal_flip(bbox, image_size, pixel_coordinate_system: PixelCoordinateSystem, pixel_definition: PixelDefinition):
    center_point = get_image_center_point(image_size, pixel_coordinate_system, pixel_definition)

    x1 = 2 * center_point[0] - bbox[2]
    x2 = 2 * center_point[0] - bbox[0]
    return (x1, bbox[1], x2, bbox[3])


def bbox_vertical_flip(bbox, image_size, pixel_coordinate_system: PixelCoordinateSystem, pixel_definition: PixelDefinition):
    center_point = get_image_center_point(image_size, pixel_coordinate_system, pixel_definition)

    y1 = 2 * center_point[1] - bbox[3]
    y2 = 2 * center_point[1] - bbox[1]
    return (bbox[0], y1, bbox[2], y2)


def bbox_flip(bbox, image_size, pixel_coordinate_system: PixelCoordinateSystem, pixel_definition: PixelDefinition):
    center_point = get_image_center_point(image_size, pixel_coordinate_system, pixel_definition)
    x1 = 2 * center_point[0] - bbox[2]
    x2 = 2 * center_point[0] - bbox[0]
    y1 = 2 * center_point[1] - bbox[3]
    y2 = 2 * center_point[1] - bbox[1]
    return (x1, y1, x2, y2)
