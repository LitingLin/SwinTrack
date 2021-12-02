from datasets.filter._common import _BaseFilter


class DataCleaning_ObjectCategory(_BaseFilter):
    def __init__(self, category_ids_to_remove, make_category_id_sequential=True):
        self.category_ids_to_remove = category_ids_to_remove
        self.make_category_id_sequential = make_category_id_sequential
