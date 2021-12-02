def bbox_rasterize_aligned(bbox):
    return tuple(int(round(v)) for v in bbox)
