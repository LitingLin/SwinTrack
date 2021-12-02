def bbox_cxcywh2xywh(bbox):
    x = bbox[0] - bbox[2] / 2
    y = bbox[1] - bbox[3] / 2
    return (x, y, bbox[2], bbox[3])
