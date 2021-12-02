from data.types.pixel_definition import PixelDefinition


def bbox_spatialize_aligned_xyxy(bbox, pixel_definition=PixelDefinition.Point):
    if pixel_definition == PixelDefinition.Point:
        return bbox
    else:
        return (bbox[0] - 0.5, bbox[1] - 0.5, bbox[2] + 0.5, bbox[3] + 0.5)


def bbox_spatialize_aligned_xyxy_pixel_as_point(bbox):
    return bbox


def bbox_spatialize_aligned_xyxy_pixel_as_region(bbox):
    return (bbox[0] - 0.5, bbox[1] - 0.5, bbox[2] + 0.5, bbox[3] + 0.5)


def bbox_spatialize_aligned_polygon(bbox):
    return bbox
