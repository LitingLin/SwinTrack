def bbox_get_intersection(bbox1, bbox2):
    inter_x1 = max(bbox1[0], bbox2[0])
    inter_y1 = max(bbox1[1], bbox2[1])
    inter_x2 = min(bbox1[2], bbox2[2])
    inter_y2 = min(bbox1[3], bbox2[3])
    if inter_x2 - inter_x1 <= 0 or inter_y2 - inter_y1 <= 0:
        return (0, 0, 0, 0)
    return (inter_x1, inter_y1, inter_x2, inter_y2)


def bbox_fit_in_boundary(bounding_box, boundary_bounding_box):
    return (max(bounding_box[0], boundary_bounding_box[0]), max(bounding_box[1], boundary_bounding_box[1]),
            min(bounding_box[2], boundary_bounding_box[2]), min(bounding_box[3], boundary_bounding_box[3]))


def bbox_fit_in_boundary_polygon(bounding_box, boundary_bounding_box):
    assert len(bounding_box) % 2 == 0
    fitted = []
    for i in range(0, len(bounding_box), 2):
        x = max(bounding_box[i], boundary_bounding_box[0])
        x = min(x, boundary_bounding_box[2])
        y = max(bounding_box[i + 1], boundary_bounding_box[1])
        y = min(y, boundary_bounding_box[3])
        fitted.append(x)
        fitted.append(y)
    return fitted
