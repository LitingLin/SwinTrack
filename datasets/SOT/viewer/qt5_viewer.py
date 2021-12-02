from datasets.SOT.dataset import SingleObjectTrackingDataset_MemoryMapped, SingleObjectTrackingDatasetSequence_MemoryMapped
from datasets.base.common.viewer.qt5_viewer import draw_object
from miscellanies.viewer.qt5_viewer import Qt5Viewer
from PyQt5.QtGui import QPixmap, QColor
import random
from miscellanies.simple_prefetcher import SimplePrefetcher


__all__ = ['SOTDatasetQt5Viewer']


class DatasetSequenceImageLoader:
    def __init__(self, sequence: SingleObjectTrackingDatasetSequence_MemoryMapped):
        self.sequence = sequence

    def __len__(self):
        return len(self.sequence)

    def __getitem__(self, index: int):
        frame = self.sequence[index]
        pixmap = QPixmap()
        assert pixmap.load(frame.get_image_path())
        return pixmap, frame


class SOTDatasetQt5Viewer:
    def __init__(self, dataset: SingleObjectTrackingDataset_MemoryMapped):
        self.dataset = dataset
        self.viewer = Qt5Viewer()
        self.viewer.set_title(self.dataset.get_name())
        self.canvas = self.viewer.get_subplot().create_canvas()

        if self.dataset.has_category_id_name_map():
            self.category_id_color_map = {}
            for category_id in self.dataset.get_category_id_name_map().keys():
                color = [random.randint(0, 255) for _ in range(3)]
                self.category_id_color_map[category_id] = QColor(color[0], color[1], color[2], int(0.5 * 255))
        else:
            self.category_id_color_map = None

        sequence_names = []
        for sequence in self.dataset:
            sequence_names.append(sequence.get_name())

        self.viewer.get_content_region().new_list(sequence_names, self._sequence_selected_callback)
        self.timer = self.viewer.new_timer()
        self.timer.set_callback(self._timer_timeout_callback)

    def _sequence_selected_callback(self, index: int):
        if index < 0:
            return
        self.sequence = SimplePrefetcher(DatasetSequenceImageLoader(self.dataset[index]))
        self._stop_timer()
        self._start_timer()

    def _start_timer(self):
        self.sequence_iter = iter(self.sequence)
        self.timer.start()

    def _timer_timeout_callback(self):
        try:
            image, frame = next(self.sequence_iter)
        except StopIteration:
            self._stop_timer()
            return
        canvas = self.canvas
        canvas.set_background(image)

        with canvas.get_painter() as painter:
            draw_object(painter, frame, frame, self.sequence.iterable.sequence, None, self.category_id_color_map, self.dataset, self.dataset)
        canvas.update()

    def _stop_timer(self):
        self.timer.stop()

    def run(self):
        return self.viewer.run_event_loop()
