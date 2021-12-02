def bbox_xywh2cxcywh(bbox):
    cx = bbox[0] + bbox[2] / 2
    cy = bbox[1] + bbox[3] / 2
    return (cx, cy, bbox[2], bbox[3])
