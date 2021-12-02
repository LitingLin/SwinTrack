def bbox_xyxy2polygon(bbox):
    return (bbox[0], bbox[1], bbox[0], bbox[3], bbox[2], bbox[3], bbox[2], bbox[1])
