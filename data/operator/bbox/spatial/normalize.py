def bbox_normalize(bbox, image_size):
    return tuple(v / image_size[0] if i % 2 == 0 else v / image_size[1] for i, v in enumerate(bbox))


def bbox_denormalize(bbox, image_size):
    return tuple(v * image_size[0] if i % 2 == 0 else v * image_size[1] for i, v in enumerate(bbox))
