import matplotlib.pyplot as plt
from PyQt5.QtWidgets import QApplication, QDialog, QVBoxLayout, QHBoxLayout, QPushButton, QSizePolicy, QLabel
from PyQt5.QtCore import QTimer, Qt, pyqtSlot, QObject
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
import matplotlib.patches as patches


class MatplotlibQt5Viewer(QObject):
    def __init__(self, argv=[]):
        super(QObject, self).__init__()
        app = QApplication(argv)

        window = QDialog()
        window.setWindowState(Qt.WindowMaximized)
        window.setWindowFlags(Qt.WindowMinimizeButtonHint | Qt.WindowMaximizeButtonHint | Qt.WindowCloseButtonHint)
        window.setWindowTitle('Viewer')

        mainLayout = QVBoxLayout()
        window.setLayout(mainLayout)
        contentLayout = QHBoxLayout()

        figure = plt.figure()
        canvas = FigureCanvas(figure)
        contentLayout.addWidget(canvas)

        toolbar = NavigationToolbar(canvas, window)
        mainLayout.addWidget(toolbar)
        mainLayout.addLayout(contentLayout)

        customLayout = QVBoxLayout()
        contentLayout.addLayout(customLayout)

        canvas.setSizePolicy(QSizePolicy.MinimumExpanding, QSizePolicy.MinimumExpanding)

        ax = figure.add_subplot()

        self.figure = figure
        self.ax = ax
        self.canvas = canvas
        self.app = app
        self.window = window
        self.customLayout = customLayout
        self.timer = QTimer()
        self.timer.timeout.connect(self._onTimerTimeOut)

    def runEventLoop(self):
        self.window.show()
        return self.app.exec_()

    def enableTimer(self):
        self.timer.start()

    def stopTimer(self):
        self.timer.stop()

    def beginDraw(self):
        self.ax.clear()

    def setWindowTitle(self, title):
        self.window.setWindowTitle(title)

    def setTimerInterval(self, msec: int):
        self.timer.setInterval(msec)

    def setTimerCallback(self, callback):
        self.callback = callback

    def addButton(self, text, callback):
        button = QPushButton()
        button.setText(text)
        button.clicked.connect(callback)
        self.customLayout.addWidget(button)
        return button

    def drawPoint(self, x_array, y_array, color=(1, 0, 0), size=10):
        self.ax.scatter(x_array, y_array, c=color, s=size)

    def drawImage(self, image):
        self.ax.imshow(image)

    def drawBoundingBox(self, bounding_box, color=(1, 0, 0), linewidth=1, linestyle='solid'):
        self.ax.add_patch(
            patches.Rectangle((bounding_box[0], bounding_box[1]), bounding_box[2], bounding_box[3], linewidth=linewidth,
                              linestyle=linestyle,
                              edgecolor=color, facecolor='none'))
    # linestyle: ['solid'|'dashed'|'dashdot'|'dotted']
    def drawBoundingBoxAndLabel(self, bounding_box, label, color=(1, 0, 0), linewidth=1, linestyle='solid'):
        self.drawBoundingBox(bounding_box, color, linewidth, linestyle)
        self.drawText(label, (bounding_box[0], bounding_box[1]), color, linestyle)

    def drawText(self, string, position, color=(1, 0, 0), edgestyle='solid'):
        self.ax.text(position[0], position[1], string, horizontalalignment='left', verticalalignment='bottom',
                     bbox={'facecolor': color, 'alpha': 0.5, 'boxstyle': 'square,pad=0', 'linewidth': 0,
                           'linestyle': edgestyle})

    @pyqtSlot()
    def _onTimerTimeOut(self):
        self.callback()

    def endDraw(self):
        self.figure.canvas.draw()

    def close(self):
        self.window.close()

    def addLabel(self, text: str):
        label = QLabel()
        label.setText(text)
        self.customLayout.addWidget(label)
        return label
