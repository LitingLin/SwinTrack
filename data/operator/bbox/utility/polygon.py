from shapely.geometry import Polygon


def get_shapely_polygon_object(bbox):
    assert len(bbox) % 2 == 0
    points = []
    for i in range(0, len(bbox), 2):
        points.append((bbox[i, i + 1]))
    return Polygon(points)
