from datasets.base.video.dataset import VideoDataset, VideoDatasetSequence, VideoDatasetFrame
from datasets.base.common.viewer.qt5_viewer import draw_object
from miscellanies.viewer.qt5_viewer import Qt5Viewer
from PyQt5.QtGui import QPixmap, QColor
from miscellanies.simple_prefetcher import SimplePrefetcher
import random

__all__ = ['VideoDatasetViewer']


class _DatasetSequenceImageLoader:
    def __init__(self, sequence: VideoDatasetSequence):
        self.sequence = sequence

    def __len__(self):
        return len(self.sequence)

    def __getitem__(self, index: int):
        frame = self.sequence[index]
        pixmap = QPixmap()
        assert pixmap.load(frame.get_image_path())

        return pixmap, frame, self.sequence


class VideoDatasetViewer:
    def __init__(self, dataset: VideoDataset):
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

        sequence_names = []
        for sequence in self.dataset:
            sequence_names.append(sequence.get_name())

        self.viewer.get_content_region().new_list(sequence_names, self._sequence_selected_callback)
        self.timer = self.viewer.new_timer()
        self.timer.set_callback(self._timer_timeout_callback)

    def _sequence_selected_callback(self, index: int):
        if index < 0:
            return
        self.sequence = SimplePrefetcher(_DatasetSequenceImageLoader(self.dataset[index]))
        self.timer.stop()
        self._start_timer()

    def _start_timer(self):
        self.sequence_iter = iter(self.sequence)
        self.timer.start()

    def _timer_timeout_callback(self):
        try:
            image, frame, sequence = next(self.sequence_iter)
        except StopIteration:
            self.timer.stop()
            return
        frame: VideoDatasetFrame = frame
        sequence: VideoDatasetSequence = sequence
        canvas = self.canvas
        canvas.set_background(image)

        with canvas.get_painter() as painter:
            for object_ in frame:
                object_category_id_accessor = object_
                if object_.has_id():
                    id_ = object_.get_id()
                    for sequence_object in sequence.get_object_iterator():
                        if sequence_object.get_id() == id_:
                            object_category_id_accessor = (object_, sequence_object)
                            break

                draw_object(painter, object_, None, object_category_id_accessor, object_, self.category_id_color_map, self.dataset, self.dataset)
        canvas.update()

    def run(self):
        return self.viewer.run_event_loop()
