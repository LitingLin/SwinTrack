from data.types.pixel_coordinate_system import PixelCoordinateSystem
from data.types.bounding_box_format import BoundingBoxFormat
from data.types.bounding_box_coordinate_system import BoundingBoxCoordinateSystem
from data.types.pixel_definition import PixelDefinition


def bbox_xyxy_transform(bbox, source_pixel_coordinate_system: PixelCoordinateSystem,
                   target_pixel_coordinate_system: PixelCoordinateSystem,
                   source_bounding_box_coordinate_system: BoundingBoxCoordinateSystem,
                   target_bounding_box_coordinate_system: BoundingBoxCoordinateSystem,
                   pixel_definition: PixelDefinition):
    if source_bounding_box_coordinate_system == BoundingBoxCoordinateSystem.Rasterized:
        if source_pixel_coordinate_system == PixelCoordinateSystem.Aligned:
            from data.operator.bbox.transform.spatialize.aligned import bbox_spatialize_aligned_xyxy
            bbox = bbox_spatialize_aligned_xyxy(bbox, pixel_definition)
        else:
            from data.operator.bbox.transform.spatialize.half_pixel_offset import bbox_spatialize_half_pixel_offset_xyxy
            bbox = bbox_spatialize_half_pixel_offset_xyxy(bbox, pixel_definition)
    # now source to float
    if source_pixel_coordinate_system != target_pixel_coordinate_system:
        if source_pixel_coordinate_system == PixelCoordinateSystem.Aligned:
            # aligned to half pixel offset
            from data.operator.bbox.transform.pixel_coordinate_system.mapping import bbox_pixel_coordinate_system_aligned_to_half_pixel_offset
            bbox = bbox_pixel_coordinate_system_aligned_to_half_pixel_offset(bbox)
        else:
            # half pixel offset to aligned
            from data.operator.bbox.transform.pixel_coordinate_system.mapping import bbox_pixel_coordinate_system_half_pixel_offset_to_aligned
            bbox = bbox_pixel_coordinate_system_half_pixel_offset_to_aligned(bbox)
    # now float & target pixel coordinate
    if target_bounding_box_coordinate_system == BoundingBoxCoordinateSystem.Rasterized:
        if target_pixel_coordinate_system == PixelCoordinateSystem.Aligned:
            from data.operator.bbox.transform.rasterize.aligned import bbox_rasterize_aligned
            bbox = bbox_rasterize_aligned(bbox)
        else:
            from data.operator.bbox.transform.rasterize.half_pixel_offset import bbox_rasterize_half_pixel_offset
            bbox = bbox_rasterize_half_pixel_offset(bbox)
    return bbox


def bbox_polygon_transform(bbox, source_pixel_coordinate_system: PixelCoordinateSystem,
                   target_pixel_coordinate_system: PixelCoordinateSystem,
                   source_bounding_box_coordinate_system: BoundingBoxCoordinateSystem,
                   target_bounding_box_coordinate_system: BoundingBoxCoordinateSystem,
                   pixel_definition: PixelDefinition=PixelDefinition.Point):
    assert pixel_definition == PixelDefinition.Point, "pixel as 1x1 square is not allowed when bounding box format is Polygon"
    if source_bounding_box_coordinate_system == BoundingBoxCoordinateSystem.Rasterized:
        if source_pixel_coordinate_system == PixelCoordinateSystem.Aligned:
            from data.operator.bbox.transform.spatialize.aligned import bbox_spatialize_aligned_polygon
            bbox = bbox_spatialize_aligned_polygon(bbox)
        else:
            from data.operator.bbox.transform.spatialize.half_pixel_offset import bbox_spatialize_half_pixel_offset_polygon
            bbox = bbox_spatialize_half_pixel_offset_polygon(bbox)
    # now bbox is spatialized
    # do pixel coordinate transform
    if source_pixel_coordinate_system != target_pixel_coordinate_system:
        if source_pixel_coordinate_system == PixelCoordinateSystem.Aligned:
            from data.operator.bbox.transform.pixel_coordinate_system.mapping import bbox_pixel_coordinate_system_aligned_to_half_pixel_offset
            bbox = bbox_pixel_coordinate_system_aligned_to_half_pixel_offset(bbox)
        else:
            from data.operator.bbox.transform.pixel_coordinate_system.mapping import bbox_pixel_coordinate_system_half_pixel_offset_to_aligned
            bbox = bbox_pixel_coordinate_system_half_pixel_offset_to_aligned(bbox)
    # now bbox is spatizlied and fit with target pixel coordinate system
    # do bounding box coordinate system transform
    if target_bounding_box_coordinate_system == BoundingBoxCoordinateSystem.Rasterized:
        if target_pixel_coordinate_system == PixelCoordinateSystem.Aligned:
            from data.operator.bbox.transform.rasterize.aligned import bbox_rasterize_aligned
            bbox = bbox_rasterize_aligned(bbox)
        else:
            from data.operator.bbox.transform.rasterize.half_pixel_offset import bbox_rasterize_half_pixel_offset
            bbox = bbox_rasterize_half_pixel_offset(bbox)
    return bbox


def bbox_rasterized_transform(bbox, source_format: BoundingBoxFormat, target_format: BoundingBoxFormat):
    if source_format == target_format:
        return bbox
    if source_format == BoundingBoxFormat.XYWH:
        from data.operator.bbox.rasterized.xywh2xyxy import bbox_xywh2xyxy
        bbox = bbox_xywh2xyxy(bbox)
    elif source_format == BoundingBoxFormat.Polygon:
        from data.operator.bbox.polygon2xyxy import bbox_polygon2xyxy
        bbox = bbox_polygon2xyxy(bbox)
    if target_format == BoundingBoxFormat.XYWH:
        from data.operator.bbox.rasterized.xyxy2xywh import bbox_xyxy2xywh
        return bbox_xyxy2xywh(bbox)
    elif target_format == BoundingBoxFormat.XYXY:
        return bbox
    elif target_format == BoundingBoxFormat.Polygon:
        from data.operator.bbox.xyxy2polygon import bbox_xyxy2polygon
        return bbox_xyxy2polygon(bbox)


def bbox_to_xyxy(bbox, format_: BoundingBoxFormat, bounding_box_coordinate_system: BoundingBoxCoordinateSystem):
    if format_ == BoundingBoxFormat.XYXY:
        return bbox
    if format_ == BoundingBoxFormat.XYWH:
        if bounding_box_coordinate_system == BoundingBoxCoordinateSystem.Rasterized:
            from data.operator.bbox.rasterized.xywh2xyxy import bbox_xywh2xyxy
            return bbox_xywh2xyxy(bbox)
        else:
            from data.operator.bbox.spatial.xywh2xyxy import bbox_xywh2xyxy
            return bbox_xywh2xyxy(bbox)
    elif format_ == BoundingBoxFormat.Polygon:
        from data.operator.bbox.polygon2xyxy import bbox_polygon2xyxy
        return bbox_polygon2xyxy(bbox)


def bbox_to_polygon(bbox, format_: BoundingBoxFormat, bounding_box_coordinate_system: BoundingBoxCoordinateSystem):
    if format_ == BoundingBoxFormat.Polygon:
        return bbox
    if format_ == BoundingBoxFormat.XYWH:
        if bounding_box_coordinate_system == BoundingBoxCoordinateSystem.Rasterized:
            from data.operator.bbox.rasterized.xywh2xyxy import bbox_xywh2xyxy
            bbox = bbox_xywh2xyxy(bbox)
        else:
            from data.operator.bbox.spatial.xywh2xyxy import bbox_xywh2xyxy
            bbox = bbox_xywh2xyxy(bbox)

    from data.operator.bbox.xyxy2polygon import bbox_xyxy2polygon
    return bbox_xyxy2polygon(bbox)


def bbox_polygon_to_any(bbox, target_format: BoundingBoxFormat, bounding_box_coordinate_system: BoundingBoxCoordinateSystem):
    if target_format == BoundingBoxFormat.Polygon:
        return bbox
    else:
        from data.operator.bbox.polygon2xyxy import bbox_polygon2xyxy
        bbox = bbox_polygon2xyxy(bbox)
        if target_format == BoundingBoxFormat.XYWH:
            if bounding_box_coordinate_system == BoundingBoxCoordinateSystem.Rasterized:
                from data.operator.bbox.rasterized.xyxy2xywh import bbox_xyxy2xywh
                return bbox_xyxy2xywh(bbox)
            else:
                from data.operator.bbox.spatial.xyxy2xywh import bbox_xyxy2xywh
                return bbox_xyxy2xywh(bbox)
        else:
            return bbox


def bbox_xyxy_to_any(bbox, target_format: BoundingBoxFormat, bounding_box_coordinate_system: BoundingBoxCoordinateSystem):
    if target_format == BoundingBoxFormat.XYXY:
        return bbox
    elif target_format == BoundingBoxFormat.XYWH:
        if bounding_box_coordinate_system == BoundingBoxCoordinateSystem.Rasterized:
            from data.operator.bbox.rasterized.xyxy2xywh import bbox_xyxy2xywh
            return bbox_xyxy2xywh(bbox)
        else:
            from data.operator.bbox.spatial.xyxy2xywh import bbox_xyxy2xywh
            return bbox_xyxy2xywh(bbox)
    else:
        from data.operator.bbox.xyxy2polygon import bbox_xyxy2polygon
        return bbox_xyxy2polygon(bbox)


def bbox_to_xywh(bbox, format_: BoundingBoxFormat, bounding_box_coordinate_system: BoundingBoxCoordinateSystem):
    if format_ == BoundingBoxFormat.XYWH:
        return bbox
    if format_ == BoundingBoxFormat.Polygon:
        from data.operator.bbox.polygon2xyxy import bbox_polygon2xyxy
        bbox = bbox_polygon2xyxy(bbox)
    if bounding_box_coordinate_system == BoundingBoxCoordinateSystem.Rasterized:
        from data.operator.bbox.rasterized.xyxy2xywh import bbox_xyxy2xywh
        return bbox_xyxy2xywh(bbox)
    else:
        from data.operator.bbox.spatial.xyxy2xywh import bbox_xyxy2xywh
        return bbox_xyxy2xywh(bbox)


def bbox_transform(bbox, source_format: BoundingBoxFormat, target_format: BoundingBoxFormat,
                   source_pixel_coordinate_system: PixelCoordinateSystem,
                   target_pixel_coordinate_system: PixelCoordinateSystem,
                   source_bounding_box_coordinate_system: BoundingBoxCoordinateSystem,
                   target_bounding_box_coordinate_system: BoundingBoxCoordinateSystem,
                   pixel_definition: PixelDefinition):
    '''
    1. polygon not allowed with PixelDefinition.Square
    2. BoundingBoxCoordinateSystem.Rasterized highway
    '''
    if source_bounding_box_coordinate_system == BoundingBoxCoordinateSystem.Rasterized and target_bounding_box_coordinate_system == BoundingBoxCoordinateSystem.Rasterized:
        # do in integer space
        return bbox_rasterized_transform(bbox, source_format, target_format)
    else:
        # do in float space
        if source_format == BoundingBoxFormat.Polygon or target_format == BoundingBoxFormat.Polygon:
            # do in polygon routine
            bbox = bbox_to_polygon(bbox, source_format, source_bounding_box_coordinate_system)
            bbox = bbox_polygon_transform(bbox, source_pixel_coordinate_system, target_pixel_coordinate_system, source_bounding_box_coordinate_system, target_bounding_box_coordinate_system, pixel_definition)
            return bbox_polygon_to_any(bbox, target_format, target_bounding_box_coordinate_system)
        else:
            # do in xyxy routine
            bbox = bbox_to_xyxy(bbox, source_format, source_bounding_box_coordinate_system)
            bbox = bbox_xyxy_transform(bbox, source_pixel_coordinate_system, target_pixel_coordinate_system, source_bounding_box_coordinate_system, target_bounding_box_coordinate_system, pixel_definition)
            return bbox_xyxy_to_any(bbox, target_format, target_bounding_box_coordinate_system)
