from data.types.pixel_definition import PixelDefinition
from data.types.pixel_coordinate_system import PixelCoordinateSystem


def get_image_bounding_box(image_size, pixel_coordinate_system: PixelCoordinateSystem, pixel_definition=PixelDefinition.Point):
    from data.operator.bbox.rasterized.xywh2xyxy import bbox_xywh2xyxy
    bbox = bbox_xywh2xyxy((0, 0, image_size[0], image_size[1]))
    if pixel_coordinate_system == PixelCoordinateSystem.Aligned:
        from data.operator.bbox.transform.spatialize.aligned import bbox_spatialize_aligned_xyxy
        return bbox_spatialize_aligned_xyxy(bbox, pixel_definition)
    else:
        from data.operator.bbox.transform.spatialize.half_pixel_offset import bbox_spatialize_half_pixel_offset_xyxy
        return bbox_spatialize_half_pixel_offset_xyxy(bbox, pixel_definition)


def bounding_box_is_intersect_with_image(bounding_box, image_size, pixel_coordinate_system: PixelCoordinateSystem, pixel_definition=PixelDefinition.Point):
    image_bounding_box = get_image_bounding_box(image_size, pixel_coordinate_system, pixel_definition)
    from data.operator.bbox.intersection import bbox_get_intersection
    from data.operator.bbox.validity import bbox_is_valid
    return bbox_is_valid(bbox_get_intersection(image_bounding_box, bounding_box))


def bounding_box_is_intersect_with_image_polygon(bounding_box, image_size, pixel_coordinate_system: PixelCoordinateSystem, pixel_definition=PixelDefinition.Point):
    from data.operator.bbox.utility.polygon import get_shapely_polygon_object
    from data.operator.bbox.xyxy2polygon import bbox_xyxy2polygon
    A = get_shapely_polygon_object(bbox_xyxy2polygon(bounding_box))
    image_bounding_box = get_image_bounding_box(image_size, pixel_coordinate_system, pixel_definition)
    B = get_shapely_polygon_object(bbox_xyxy2polygon(image_bounding_box))
    return A.intersection(B).area > 0


def get_image_center_point(image_size, pixel_coordinate_system: PixelCoordinateSystem, pixel_definition=PixelDefinition.Point):
    image_bounding_box = get_image_bounding_box(image_size, pixel_coordinate_system, pixel_definition)
    from data.operator.bbox.spatial.center import bbox_get_center_point
    return bbox_get_center_point(image_bounding_box)


def bounding_box_fit_in_image_boundary(bbox, image_size, pixel_coordinate_system: PixelCoordinateSystem, pixel_definition=PixelDefinition.Point):
    image_bounding_box = get_image_bounding_box(image_size, pixel_coordinate_system, pixel_definition)
    from data.operator.bbox.intersection import bbox_fit_in_boundary
    return bbox_fit_in_boundary(bbox, image_bounding_box)


def bounding_box_fit_in_image_boundary_polygon(bbox, image_size, pixel_coordinate_system: PixelCoordinateSystem, pixel_definition=PixelDefinition.Point):
    image_bounding_box = get_image_bounding_box(image_size, pixel_coordinate_system, pixel_definition)
    from data.operator.bbox.intersection import bbox_fit_in_boundary_polygon
    return bbox_fit_in_boundary_polygon(bbox, image_bounding_box)
