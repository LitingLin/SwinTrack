def bbox_compute_iou(bbox_a, bbox_b):
    from data.operator.bbox.intersection import bbox_get_intersection
    intersection = bbox_get_intersection(bbox_a, bbox_b)
    area_a = (bbox_a[2] - bbox_a[0]) * (bbox_a[3] - bbox_a[1])
    area_b = (bbox_b[2] - bbox_b[0]) * (bbox_b[3] - bbox_b[1])
    area_intersection = (intersection[2] - intersection[0]) * (intersection[3] - intersection[1])
    return area_intersection / (area_a + area_b - area_intersection)
