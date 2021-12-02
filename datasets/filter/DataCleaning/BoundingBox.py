from datasets.filter._common import _BaseFilter


class DataCleaning_BoundingBox(_BaseFilter):
    def __init__(self, fit_in_image_size:bool=False, update_validity:bool=False, remove_invalid_objects=False, remove_empty_objects=False):
        self.fit_in_image_size = fit_in_image_size
        self.update_validity = update_validity
        self.remove_invalid_objects = remove_invalid_objects
        self.remove_empty_objects = remove_empty_objects
