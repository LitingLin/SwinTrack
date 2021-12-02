from enum import Enum, auto


class BoundingBoxFormat(Enum):
    r'''
    XYWH: [x, y, w, h]
    XYXY: [x1, y1, x2, y2]
    Polygon: [x1, y1, x2, y2, x3, y3, x4, y4]
    '''
    XYWH = auto()
    XYXY = auto()
    Polygon = auto()
    CXCYWH = auto()
