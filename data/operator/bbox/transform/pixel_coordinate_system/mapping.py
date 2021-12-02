def bbox_pixel_coordinate_system_aligned_to_half_pixel_offset(bbox):
    return tuple(v + 0.5 for v in bbox)


def bbox_pixel_coordinate_system_half_pixel_offset_to_aligned(bbox):
    return tuple(v - 0.5 for v in bbox)
