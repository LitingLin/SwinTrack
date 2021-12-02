from PyQt5.QtGui import QImage
import numpy as np


def qimage_to_numpy_rgb888(qimage: QImage):
    if qimage.format() != QImage.Format_RGB888:
        qimage = qimage.convertToFormat(QImage.Format_RGB888)
    b = qimage.bits()
    b.setsize(qimage.width() * qimage.height() * 3)
    return np.frombuffer(b, dtype=np.uint8).reshape(qimage.height(), qimage.width(), 3)


def numpy_rgb888_to_qimage(image: np.ndarray):
    image = np.ascontiguousarray(image)
    h, w, c = image.shape
    assert c == 3
    qimage = QImage(image.data, w, h, 3 * w, QImage.Format_RGB888)
    return qimage
