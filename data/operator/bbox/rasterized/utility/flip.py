def bbox_horizontal_flip(bbox, image_size):
    w = image_size[0] - 1
    x1 = w - bbox[2]
    x2 = w - bbox[0]
    return (x1, bbox[1], x2, bbox[3])


def bbox_vertical_flip(bbox, image_size):
    h = image_size[1] - 1
    y1 = h - bbox[3]
    y2 = h - bbox[1]
    return (bbox[0], y1, bbox[2], y2)


def bbox_flip(bbox, image_size):
    w = image_size[0] - 1
    x1 = w - bbox[2]
    x2 = w - bbox[0]
    h = image_size[1] - 1
    y1 = h - bbox[3]
    y2 = h - bbox[1]
    return (x1, y1, x2, y2)
