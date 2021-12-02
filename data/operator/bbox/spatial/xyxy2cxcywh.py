def bbox_xyxy2cxcywh(bbox):
    from .xyxy2xywh import bbox_xyxy2xywh
    from .xywh2cxcywh import bbox_xywh2cxcywh
    return bbox_xywh2cxcywh(bbox_xyxy2xywh(bbox))
