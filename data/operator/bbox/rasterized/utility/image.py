def get_image_bounding_box(image_size):
    from data.operator.bbox.rasterized.xywh2xyxy import bbox_xywh2xyxy
    return bbox_xywh2xyxy((0, 0, image_size[0], image_size[1]))


def bounding_box_is_intersect_with_image(bounding_box, image_size):
    image_bounding_box = get_image_bounding_box(image_size)
    from data.operator.bbox.intersection import bbox_get_intersection
    from data.operator.bbox.validity import bbox_is_valid
    return bbox_is_valid(bbox_get_intersection(image_bounding_box, bounding_box))


def bounding_box_is_intersect_with_image_polygon(bounding_box, image_size):
    from data.operator.bbox.utility.polygon import get_shapely_polygon_object
    from data.operator.bbox.xyxy2polygon import bbox_xyxy2polygon
    A = get_shapely_polygon_object(bbox_xyxy2polygon(bounding_box))
    image_bounding_box = get_image_bounding_box(image_size)
    B = get_shapely_polygon_object(bbox_xyxy2polygon(image_bounding_box))
    return A.intersection(B).area > 0


def bounding_box_fit_in_image_boundary(bbox, image_size):
    image_bounding_box = get_image_bounding_box(image_size)
    from data.operator.bbox.intersection import bbox_fit_in_boundary
    return bbox_fit_in_boundary(bbox, image_bounding_box)


def bounding_box_fit_in_image_boundary_polygon(bbox, image_size):
    image_bounding_box = get_image_bounding_box(image_size)
    from data.operator.bbox.intersection import bbox_fit_in_boundary_polygon
    return bbox_fit_in_boundary_polygon(bbox, image_bounding_box)
