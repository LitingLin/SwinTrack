from ._common import _BaseFilter
import numpy as np


class SortByImageRatio(_BaseFilter):
    def __init__(self, descending=False):
        self.descending = descending

    def __call__(self, sizes):
        sizes = np.array(sizes)

        if self.descending:
            ratio = sizes[:, 0] / sizes[:, 1]
        else:
            ratio = sizes[:, 1] / sizes[:, 0]

        indices = ratio.argsort()
        return indices
