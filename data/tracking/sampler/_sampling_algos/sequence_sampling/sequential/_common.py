from data.operator.bbox.validity import bbox_is_valid
from data.operator.bbox.spatial.utility.half_pixel_offset.image import bounding_box_is_intersect_with_image


def _check_bounding_box_validity(bounding_box, bounding_box_validity_flag, image_size):
    if bounding_box_validity_flag is not None and not bounding_box_validity_flag:
        bounding_box = None
    elif bounding_box is not None:
        assert bbox_is_valid(bounding_box) and bounding_box_is_intersect_with_image(bounding_box, image_size)
    return bounding_box
