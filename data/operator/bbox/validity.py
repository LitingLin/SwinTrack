def bbox_is_valid(bbox):
    return bbox[0] < bbox[2] and bbox[1] < bbox[3]


def bbox_is_valid_xywh(bbox):
    return bbox[2] > 0 and bbox[3] > 0
