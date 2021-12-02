def draw_bbox_(canvas, bbox, value):
    canvas[bbox[1]: bbox[3] + 1, bbox[0]] = value
    canvas[bbox[1]: bbox[3] + 1, bbox[2]] = value
    canvas[bbox[1], bbox[0]: bbox[2] + 1] = value
    canvas[bbox[3], bbox[0]: bbox[2] + 1] = value
