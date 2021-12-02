from datasets.DET.dataset import DetectionDataset_MemoryMapped
from datasets.base.common.viewer.qt5_viewer import draw_object
from miscellanies.viewer.qt5_viewer import Qt5Viewer
from PyQt5.QtGui import QPixmap, QColor
import random


class DetectionDatasetQt5Viewer:
    def __init__(self, dataset: DetectionDataset_MemoryMapped):
        self.dataset = dataset
        self.viewer = Qt5Viewer()
        self.canvas = self.viewer.get_subplot().create_canvas()

        if self.dataset.has_category_id_name_map():
            self.category_id_color_map = {}
            for category_id in self.dataset.get_category_id_name_map().keys():
                color = [random.randint(0, 255) for _ in range(3)]
                self.category_id_color_map[category_id] = QColor(color[0], color[1], color[2], int(0.5 * 255))
        else:
            self.category_id_color_map = None

        image_names = []
        for index in range(len(self.dataset)):
            image_names.append(str(index))

        self.viewer.get_content_region().new_list(image_names, self._image_selected_callback)

    def _image_selected_callback(self, index: int):
        if index < 0:
            return
        image = self.dataset[index]
        pixmap = QPixmap()
        assert pixmap.load(image.get_image_path())
        canvas = self.canvas
        canvas.set_background(pixmap)

        if len(image) > 0:
            with canvas.get_painter() as painter:
                for object_ in image:
                    draw_object(painter, object_, object_, object_, None, self.category_id_color_map,
                                self.dataset, self.dataset)
        canvas.update()

    def run(self):
        return self.viewer.run_event_loop()
