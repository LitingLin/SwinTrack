from data.types.pixel_coordinate_system import PixelCoordinateSystem
from data.types.bounding_box_format import BoundingBoxFormat
from data.types.bounding_box_coordinate_system import BoundingBoxCoordinateSystem
from data.types.pixel_definition import PixelDefinition


def _common_routine(bounding_box, image_size, bounding_box_format: BoundingBoxFormat,
                    pixel_coordinate_system: PixelCoordinateSystem,
                    bounding_box_coordinate_system: BoundingBoxCoordinateSystem,
                    pixel_definition, rasterized_xyxy_func, rasterized_polygon_func,
                    spatial_xyxy_func, spatial_polygon_func):
    if bounding_box_coordinate_system == BoundingBoxCoordinateSystem.Rasterized:
        if bounding_box_format == BoundingBoxFormat.XYWH or bounding_box_format == BoundingBoxFormat.XYXY:
            if bounding_box_format == BoundingBoxFormat.XYWH:
                from data.operator.bbox.rasterized.xywh2xyxy import bbox_xywh2xyxy
                bounding_box = bbox_xywh2xyxy(bounding_box)
            return rasterized_xyxy_func(bounding_box, image_size)
        else:
            return rasterized_polygon_func(bounding_box, image_size)
    else:
        if bounding_box_format == BoundingBoxFormat.XYWH or bounding_box_format == BoundingBoxFormat.XYXY:
            if bounding_box_format == BoundingBoxFormat.XYWH:
                from data.operator.bbox.rasterized.xywh2xyxy import bbox_xywh2xyxy
                bounding_box = bbox_xywh2xyxy(bounding_box)
            return spatial_xyxy_func(bounding_box, image_size, pixel_coordinate_system, pixel_definition)
        else:
            return spatial_polygon_func(bounding_box, image_size, pixel_coordinate_system, pixel_definition)


def bounding_box_is_intersect_with_image(bounding_box, image_size, bounding_box_format: BoundingBoxFormat,
                                         pixel_coordinate_system: PixelCoordinateSystem,
                                         bounding_box_coordinate_system: BoundingBoxCoordinateSystem,
                                         pixel_definition: PixelDefinition = PixelDefinition.Point):
    import data.operator.bbox.rasterized.utility.image
    import data.operator.bbox.spatial.utility.image
    return _common_routine(bounding_box, image_size, bounding_box_format, pixel_coordinate_system,
                           bounding_box_coordinate_system, pixel_definition,
                           data.operator.bbox.rasterized.utility.image.bounding_box_is_intersect_with_image,
                           data.operator.bbox.rasterized.utility.image.bounding_box_is_intersect_with_image_polygon,
                           data.operator.bbox.spatial.utility.image.bounding_box_is_intersect_with_image,
                           data.operator.bbox.spatial.utility.image.bounding_box_is_intersect_with_image_polygon)


def bounding_box_fit_in_image_boundary(bounding_box, image_size, bounding_box_format: BoundingBoxFormat,
                                       pixel_coordinate_system: PixelCoordinateSystem,
                                       bounding_box_coordinate_system: BoundingBoxCoordinateSystem,
                                       pixel_definition: PixelDefinition = PixelDefinition.Point):
    import data.operator.bbox.rasterized.utility.image
    import data.operator.bbox.spatial.utility.image
    return _common_routine(bounding_box, image_size, bounding_box_format, pixel_coordinate_system,
                           bounding_box_coordinate_system, pixel_definition,
                           data.operator.bbox.rasterized.utility.image.bounding_box_fit_in_image_boundary,
                           data.operator.bbox.rasterized.utility.image.bounding_box_fit_in_image_boundary_polygon,
                           data.operator.bbox.spatial.utility.image.bounding_box_fit_in_image_boundary,
                           data.operator.bbox.spatial.utility.image.bounding_box_fit_in_image_boundary_polygon)
