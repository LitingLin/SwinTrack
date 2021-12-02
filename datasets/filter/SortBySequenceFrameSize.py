from ._common import _BaseFilter
import numpy as np


class SortBySequenceFrameSize(_BaseFilter):
    def __init__(self, descending=False):
        self.descending = descending

    def __call__(self, sizes):
        sizes = np.array(sizes)
        if self.descending:
            indices = (-sizes).argsort()
        else:
            indices = sizes.argsort()
        return indices
