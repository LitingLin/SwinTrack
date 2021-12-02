from PyQt5.QtCore import Qt
from PyQt5.QtGui import QColor, QPen
from data.operator.bbox.transform.transform import bbox_transform
from data.types.bounding_box_format import BoundingBoxFormat
from data.types.pixel_coordinate_system import PixelCoordinateSystem
from data.types.bounding_box_coordinate_system import BoundingBoxCoordinateSystem
from data.types.pixel_definition import PixelDefinition


def draw_object(painter, bounding_box_accessor, bounding_box_validity_flag_accessor, category_id_accessor,
                object_id_accessor, category_id_color_map, category_id_name_map_accessor, context_accessor):
    if category_id_name_map_accessor.has_category_id_name_map():
        category_id_name_map = category_id_name_map_accessor.get_category_id_name_map()
    else:
        category_id_name_map = None
    if bounding_box_accessor.has_bounding_box():
        category_id = None
        if category_id_accessor is not None:
            if isinstance(category_id_accessor, (list, tuple)):
                for c_category_id_accessor in category_id_accessor:
                    if c_category_id_accessor.has_category_id():
                        category_id = c_category_id_accessor.get_category_id()
                        break
            else:
                if category_id_accessor.has_category_id():
                    category_id = category_id_accessor.get_category_id()
        bounding_box = bounding_box_accessor.get_bounding_box()
        if bounding_box_validity_flag_accessor is not None:
            if bounding_box_validity_flag_accessor.has_bounding_box_validity_flag():
                bounding_box_validity_flag = bounding_box_validity_flag_accessor.get_bounding_box_validity_flag()
            else:
                bounding_box_validity_flag = True
        else:
            bounding_box, bounding_box_validity_flag = bounding_box
            if bounding_box_validity_flag is None:
                bounding_box_validity_flag = True
        if category_id is None or category_id_color_map is None:
            color = QColor(255, 0, 0, int(0.5 * 255))
        else:
            color = category_id_color_map[category_id]
        pen = QPen(color)
        if bounding_box_validity_flag is False:
            pen.setStyle(Qt.DashDotDotLine)
        painter.set_pen(pen)
        bounding_box_format: BoundingBoxFormat = context_accessor.get_bounding_box_format()
        if bounding_box_format != BoundingBoxFormat.Polygon:
            bounding_box = bbox_transform(bounding_box, bounding_box_format, BoundingBoxFormat.XYWH,
                                          context_accessor.get_pixel_coordinate_system(), PixelCoordinateSystem.Aligned,
                                          context_accessor.get_bounding_box_coordinate_system(),
                                          BoundingBoxCoordinateSystem.Rasterized,
                                          PixelDefinition.Point)
            painter.draw_rect(bounding_box)
        else:
            bounding_box = bbox_transform(bounding_box, bounding_box_format, BoundingBoxFormat.Polygon,
                                          context_accessor.get_pixel_coordinate_system(), PixelCoordinateSystem.Aligned,
                                          context_accessor.get_bounding_box_coordinate_system(),
                                          BoundingBoxCoordinateSystem.Rasterized,
                                          PixelDefinition.Point)
            painter.draw_polygon(bounding_box)

        label_string = []
        if object_id_accessor is not None and ((not hasattr(object_id_accessor, 'has_id') and hasattr(
                object_id_accessor, 'get_id')) or object_id_accessor.has_id()):
            label_string.append(str(object_id_accessor.get_id()))
        if not (category_id is None or category_id_color_map is None or category_id_name_map is None):
            label_string.append(category_id_name_map[category_id])
        painter.draw_label('-'.join(label_string), (bounding_box[0], bounding_box[1]), color)
