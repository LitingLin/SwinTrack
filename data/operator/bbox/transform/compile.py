from data.types.pixel_coordinate_system import PixelCoordinateSystem
from data.types.bounding_box_format import BoundingBoxFormat
from data.types.bounding_box_coordinate_system import BoundingBoxCoordinateSystem
from data.types.pixel_definition import PixelDefinition


def _compile_bbox_xyxy_transform(commands: list, source_pixel_coordinate_system: PixelCoordinateSystem,
                                 target_pixel_coordinate_system: PixelCoordinateSystem,
                                 source_bounding_box_coordinate_system: BoundingBoxCoordinateSystem,
                                 target_bounding_box_coordinate_system: BoundingBoxCoordinateSystem,
                                 pixel_definition: PixelDefinition):
    if source_bounding_box_coordinate_system == BoundingBoxCoordinateSystem.Rasterized:
        if source_pixel_coordinate_system == PixelCoordinateSystem.Aligned:
            if pixel_definition == PixelDefinition.Point:
                from data.operator.bbox.transform.spatialize.aligned import bbox_spatialize_aligned_xyxy_pixel_as_point
                commands.append(bbox_spatialize_aligned_xyxy_pixel_as_point)
            else:
                from data.operator.bbox.transform.spatialize.aligned import bbox_spatialize_aligned_xyxy_pixel_as_region
                commands.append(bbox_spatialize_aligned_xyxy_pixel_as_region)
        else:
            if pixel_definition == PixelDefinition.Point:
                from data.operator.bbox.transform.spatialize.half_pixel_offset import \
                    bbox_spatialize_half_pixel_offset_xyxy_pixel_as_point
                commands.append(bbox_spatialize_half_pixel_offset_xyxy_pixel_as_point)
            else:
                from data.operator.bbox.transform.spatialize.half_pixel_offset import \
                    bbox_spatialize_half_pixel_offset_xyxy_pixel_as_region
                commands.append(bbox_spatialize_half_pixel_offset_xyxy_pixel_as_region)
    # now source to float
    if source_pixel_coordinate_system != target_pixel_coordinate_system:
        if source_pixel_coordinate_system == PixelCoordinateSystem.Aligned:
            # aligned to half pixel offset
            from data.operator.bbox.transform.pixel_coordinate_system.mapping import \
                bbox_pixel_coordinate_system_aligned_to_half_pixel_offset
            commands.append(bbox_pixel_coordinate_system_aligned_to_half_pixel_offset)
        else:
            # half pixel offset to aligned
            from data.operator.bbox.transform.pixel_coordinate_system.mapping import \
                bbox_pixel_coordinate_system_half_pixel_offset_to_aligned
            commands.append(bbox_pixel_coordinate_system_half_pixel_offset_to_aligned)
    # now float & target pixel coordinate
    if target_bounding_box_coordinate_system == BoundingBoxCoordinateSystem.Rasterized:
        if target_pixel_coordinate_system == PixelCoordinateSystem.Aligned:
            from data.operator.bbox.transform.rasterize.aligned import bbox_rasterize_aligned
            commands.append(bbox_rasterize_aligned)
        else:
            from data.operator.bbox.transform.rasterize.half_pixel_offset import bbox_rasterize_half_pixel_offset
            commands.append(bbox_rasterize_half_pixel_offset)


def _compile_bbox_polygon_transform(commands: list, source_pixel_coordinate_system: PixelCoordinateSystem,
                                    target_pixel_coordinate_system: PixelCoordinateSystem,
                                    source_bounding_box_coordinate_system: BoundingBoxCoordinateSystem,
                                    target_bounding_box_coordinate_system: BoundingBoxCoordinateSystem,
                                    pixel_definition: PixelDefinition = PixelDefinition.Point):
    assert pixel_definition == PixelDefinition.Point, "pixel as 1x1 square is not allowed when bounding box format is Polygon"
    if source_bounding_box_coordinate_system == BoundingBoxCoordinateSystem.Rasterized:
        if source_pixel_coordinate_system == PixelCoordinateSystem.Aligned:
            from data.operator.bbox.transform.spatialize.aligned import bbox_spatialize_aligned_polygon
            commands.append(bbox_spatialize_aligned_polygon)
        else:
            from data.operator.bbox.transform.spatialize.half_pixel_offset import \
                bbox_spatialize_half_pixel_offset_polygon
            commands.append(bbox_spatialize_half_pixel_offset_polygon)
    # now bbox is spatialized
    # do pixel coordinate transform
    if source_pixel_coordinate_system != target_pixel_coordinate_system:
        if source_pixel_coordinate_system == PixelCoordinateSystem.Aligned:
            from data.operator.bbox.transform.pixel_coordinate_system.mapping import \
                bbox_pixel_coordinate_system_aligned_to_half_pixel_offset
            commands.append(bbox_pixel_coordinate_system_aligned_to_half_pixel_offset)
        else:
            from data.operator.bbox.transform.pixel_coordinate_system.mapping import \
                bbox_pixel_coordinate_system_half_pixel_offset_to_aligned
            commands.append(bbox_pixel_coordinate_system_half_pixel_offset_to_aligned)
    # now bbox is spatizlied and fit with target pixel coordinate system
    # do bounding box coordinate system transform
    if target_bounding_box_coordinate_system == BoundingBoxCoordinateSystem.Rasterized:
        if target_pixel_coordinate_system == PixelCoordinateSystem.Aligned:
            from data.operator.bbox.transform.rasterize.aligned import bbox_rasterize_aligned
            commands.append(bbox_rasterize_aligned)
        else:
            from data.operator.bbox.transform.rasterize.half_pixel_offset import bbox_rasterize_half_pixel_offset
            commands.append(bbox_rasterize_half_pixel_offset)


def _compile_bbox_rasterized_transform(commands: list, source_format: BoundingBoxFormat,
                                       target_format: BoundingBoxFormat):
    if source_format == target_format:
        return
    if source_format == BoundingBoxFormat.XYWH:
        from data.operator.bbox.rasterized.xywh2xyxy import bbox_xywh2xyxy
        commands.append(bbox_xywh2xyxy)
    elif source_format == BoundingBoxFormat.Polygon:
        from data.operator.bbox.polygon2xyxy import bbox_polygon2xyxy
        commands.append(bbox_polygon2xyxy)
    if target_format == BoundingBoxFormat.XYWH:
        from data.operator.bbox.rasterized.xyxy2xywh import bbox_xyxy2xywh
        commands.append(bbox_xyxy2xywh)
    elif target_format == BoundingBoxFormat.Polygon:
        from data.operator.bbox.xyxy2polygon import bbox_xyxy2polygon
        commands.append(bbox_xyxy2polygon)


def _compile_bbox_to_xyxy(commands: list, format_: BoundingBoxFormat,
                          bounding_box_coordinate_system: BoundingBoxCoordinateSystem):
    if format_ == BoundingBoxFormat.XYXY:
        return
    if format_ == BoundingBoxFormat.XYWH:
        if bounding_box_coordinate_system == BoundingBoxCoordinateSystem.Rasterized:
            from data.operator.bbox.rasterized.xywh2xyxy import bbox_xywh2xyxy
            commands.append(bbox_xywh2xyxy)
        else:
            from data.operator.bbox.spatial.xywh2xyxy import bbox_xywh2xyxy
            commands.append(bbox_xywh2xyxy)
    elif format_ == BoundingBoxFormat.Polygon:
        from data.operator.bbox.polygon2xyxy import bbox_polygon2xyxy
        commands.append(bbox_polygon2xyxy)


def _compile_bbox_to_polygon(commands: list, format_: BoundingBoxFormat,
                             bounding_box_coordinate_system: BoundingBoxCoordinateSystem):
    if format_ == BoundingBoxFormat.Polygon:
        return
    if format_ == BoundingBoxFormat.XYWH:
        if bounding_box_coordinate_system == BoundingBoxCoordinateSystem.Rasterized:
            from data.operator.bbox.rasterized.xywh2xyxy import bbox_xywh2xyxy
            commands.append(bbox_xywh2xyxy)
        else:
            from data.operator.bbox.spatial.xywh2xyxy import bbox_xywh2xyxy
            commands.append(bbox_xywh2xyxy)

    from data.operator.bbox.xyxy2polygon import bbox_xyxy2polygon
    commands.append(bbox_xyxy2polygon)


def _compile_bbox_polygon_to_any(commands: list, target_format: BoundingBoxFormat,
                                 bounding_box_coordinate_system: BoundingBoxCoordinateSystem):
    if target_format == BoundingBoxFormat.Polygon:
        return
    else:
        from data.operator.bbox.polygon2xyxy import bbox_polygon2xyxy
        commands.append(bbox_polygon2xyxy)
        if target_format == BoundingBoxFormat.XYWH:
            if bounding_box_coordinate_system == BoundingBoxCoordinateSystem.Rasterized:
                from data.operator.bbox.rasterized.xyxy2xywh import bbox_xyxy2xywh
                commands.append(bbox_xyxy2xywh)
            else:
                from data.operator.bbox.spatial.xyxy2xywh import bbox_xyxy2xywh
                commands.append(bbox_xyxy2xywh)


def _compile_bbox_xyxy_to_any(commands: list, target_format: BoundingBoxFormat,
                              bounding_box_coordinate_system: BoundingBoxCoordinateSystem):
    if target_format == BoundingBoxFormat.XYXY:
        return
    elif target_format == BoundingBoxFormat.XYWH:
        if bounding_box_coordinate_system == BoundingBoxCoordinateSystem.Rasterized:
            from data.operator.bbox.rasterized.xyxy2xywh import bbox_xyxy2xywh
            commands.append(bbox_xyxy2xywh)
        else:
            from data.operator.bbox.spatial.xyxy2xywh import bbox_xyxy2xywh
            commands.append(bbox_xyxy2xywh)
    else:
        from data.operator.bbox.xyxy2polygon import bbox_xyxy2polygon
        commands.append(bbox_xyxy2polygon)


def _compile_bbox_to_xywh(commands: list, format_: BoundingBoxFormat,
                          bounding_box_coordinate_system: BoundingBoxCoordinateSystem):
    if format_ == BoundingBoxFormat.XYWH:
        return
    if format_ == BoundingBoxFormat.Polygon:
        from data.operator.bbox.polygon2xyxy import bbox_polygon2xyxy
        commands.append(bbox_polygon2xyxy)
    if bounding_box_coordinate_system == BoundingBoxCoordinateSystem.Rasterized:
        from data.operator.bbox.rasterized.xyxy2xywh import bbox_xyxy2xywh
        commands.append(bbox_xyxy2xywh)
    else:
        from data.operator.bbox.spatial.xyxy2xywh import bbox_xyxy2xywh
        commands.append(bbox_xyxy2xywh)


def compile_bbox_transform(source_format: BoundingBoxFormat, target_format: BoundingBoxFormat,
                           source_pixel_coordinate_system: PixelCoordinateSystem,
                           target_pixel_coordinate_system: PixelCoordinateSystem,
                           source_bounding_box_coordinate_system: BoundingBoxCoordinateSystem,
                           target_bounding_box_coordinate_system: BoundingBoxCoordinateSystem,
                           pixel_definition: PixelDefinition):
    commands = []
    if source_bounding_box_coordinate_system == BoundingBoxCoordinateSystem.Rasterized and target_bounding_box_coordinate_system == BoundingBoxCoordinateSystem.Rasterized:
        # do in integer space
        _compile_bbox_rasterized_transform(commands, source_format, target_format)
    else:
        # do in float space
        if source_format == BoundingBoxFormat.Polygon or target_format == BoundingBoxFormat.Polygon:
            # do in polygon routine
            _compile_bbox_to_polygon(commands, source_format, source_bounding_box_coordinate_system)
            _compile_bbox_polygon_transform(commands, source_pixel_coordinate_system, target_pixel_coordinate_system,
                                            source_bounding_box_coordinate_system,
                                            target_bounding_box_coordinate_system, pixel_definition)
            _compile_bbox_polygon_to_any(commands, target_format, target_bounding_box_coordinate_system)
        else:
            # do in xyxy routine
            _compile_bbox_to_xyxy(commands, source_format, source_bounding_box_coordinate_system)
            _compile_bbox_xyxy_transform(commands, source_pixel_coordinate_system, target_pixel_coordinate_system,
                                         source_bounding_box_coordinate_system, target_bounding_box_coordinate_system,
                                         pixel_definition)
            _compile_bbox_xyxy_to_any(commands, target_format, target_bounding_box_coordinate_system)

    class _BoundingBoxConverter:
        def __init__(self, commands_):
            self.commands = commands_

        def __call__(self, bbox):
            for command in self.commands:
                bbox = command(bbox)
            return bbox
    return _BoundingBoxConverter(commands)
