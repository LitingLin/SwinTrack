def bbox_polygon2xyxy(bbox):
    xs = bbox[0::2]
    ys = bbox[1::2]
    return (min(xs), min(ys), max(xs), max(ys))
