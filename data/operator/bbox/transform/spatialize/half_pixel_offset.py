from data.types.pixel_definition import PixelDefinition


def bbox_spatialize_half_pixel_offset_xyxy(bbox, pixel_definition=PixelDefinition.Point):
    if pixel_definition == PixelDefinition.Point:
        return (bbox[0] + 0.5, bbox[1] + 0.5, bbox[2] + 0.5, bbox[3] + 0.5)
    else:
        return (bbox[0], bbox[1], bbox[2] + 1, bbox[3] + 1)


def bbox_spatialize_half_pixel_offset_xyxy_pixel_as_point(bbox):
    return (bbox[0] + 0.5, bbox[1] + 0.5, bbox[2] + 0.5, bbox[3] + 0.5)


def bbox_spatialize_half_pixel_offset_xyxy_pixel_as_region(bbox):
    return (bbox[0], bbox[1], bbox[2] + 1, bbox[3] + 1)


def bbox_spatialize_half_pixel_offset_polygon(bbox):
    return (v + 0.5 for v in bbox)
