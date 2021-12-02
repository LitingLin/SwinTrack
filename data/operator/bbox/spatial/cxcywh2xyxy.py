def bbox_cxcywh2xyxy(bbox):
    from .cxcywh2xywh import bbox_cxcywh2xywh
    from .xywh2xyxy import bbox_xywh2xyxy
    return bbox_xywh2xyxy(bbox_cxcywh2xywh(bbox))
