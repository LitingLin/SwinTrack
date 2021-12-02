import data.operator.bbox.spatial.utility.image
from data.types.pixel_coordinate_system import PixelCoordinateSystem


def get_image_center_point(image_size):
    return data.operator.bbox.spatial.utility.image.get_image_center_point(image_size, PixelCoordinateSystem.Aligned)


def get_image_bounding_box(image_size):
    return data.operator.bbox.spatial.utility.image.get_image_bounding_box(image_size, PixelCoordinateSystem.Aligned)


def bbox_scale_with_image_resize(bbox, old_size, new_size):
    old_bounding_box = get_image_bounding_box(old_size)
    new_bounding_box = get_image_bounding_box(new_size)
    scale = ((new_bounding_box[2] - new_bounding_box[0]) / (old_bounding_box[2] - old_bounding_box[0]),
             (new_bounding_box[3] - new_bounding_box[1]) / (old_bounding_box[3] - old_bounding_box[1]))

    return tuple(v * scale[0] if i % 2 == 0 else v * scale[1] for i, v in enumerate(bbox))


def bounding_box_fit_in_image_boundary(bbox, image_size):
    image_bounding_box = get_image_bounding_box(image_size)
    from data.operator.bbox.intersection import bbox_fit_in_boundary
    return bbox_fit_in_boundary(bbox, image_bounding_box)


def bounding_box_is_intersect_with_image(bounding_box, image_size):
    image_bounding_box = get_image_bounding_box(image_size)
    from data.operator.bbox.intersection import bbox_get_intersection
    from data.operator.bbox.validity import bbox_is_valid
    return bbox_is_valid(bbox_get_intersection(image_bounding_box, bounding_box))
