from datasets.base.common.constructor import BaseDatasetConstructorGenerator, BaseImageDatasetConstructor, BaseDatasetImageConstructorGenerator, BaseDatasetImageConstructor
from datasets.base.common.operator.bounding_box import set_bounding_box_


class DetectionDatasetObjectConstructor:
    def __init__(self, object_: dict, category_id_name_map: dict, context):
        self.object_ = object_
        self.category_id_name_map = category_id_name_map
        self.context = context

    def set_bounding_box(self, bounding_box, validity=None, dtype=None):
        set_bounding_box_(self.object_, bounding_box, validity, dtype, self.context)

    def set_category_id(self, category_id):
        assert category_id in self.category_id_name_map
        self.object_['category_id'] = category_id

    def set_attribute(self, name: str, value):
        self.object_[name] = value

    def merge_attributes(self, attributes):
        for key, value in attributes.items():
            self.object_[key] = value


class DetectionDatasetObjectConstructorGenerator:
    def __init__(self, object_: dict, category_id_name_map: dict, context):
        self.object_ = object_
        self.category_id_name_map = category_id_name_map
        self.context = context

    def __enter__(self):
        return DetectionDatasetObjectConstructor(self.object_, self.category_id_name_map, self.context)

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass


class DetectionDatasetImageConstructor(BaseDatasetImageConstructor):
    def __init__(self, image: dict, root_path: str, category_id_name_map: dict, context):
        super(DetectionDatasetImageConstructor, self).__init__(image, root_path, context, category_id_name_map)

    def new_object(self):
        object_ = {}
        if 'objects' not in self.image:
            self.image['objects'] = []
        self.image['objects'].append(object_)
        return DetectionDatasetObjectConstructorGenerator(object_, self.category_id_name_map, self.context)


class DetectionDatasetImageConstructorGenerator(BaseDatasetImageConstructorGenerator):
    def __init__(self, image: dict, root_path: str, category_id_name_map: dict, context):
        super(DetectionDatasetImageConstructorGenerator, self).__init__(context)
        self.image = image
        self.root_path = root_path
        self.category_id_name_map = category_id_name_map

    def __enter__(self):
        return DetectionDatasetImageConstructor(self.image, self.root_path, self.category_id_name_map, self.context)


class DetectionDatasetConstructor(BaseImageDatasetConstructor):
    def __init__(self, dataset: dict, root_path: str, version: int, context):
        super(DetectionDatasetConstructor, self).__init__(dataset, root_path, version, context)

    def new_image(self):
        image = {}
        self.dataset['images'].append(image)
        if 'category_id_name_map' in self.dataset:
            category_id_name_map = self.dataset['category_id_name_map']
        else:
            category_id_name_map = None
        return DetectionDatasetImageConstructorGenerator(image, self.root_path, category_id_name_map, self.context)


class DetectionDatasetConstructorGenerator(BaseDatasetConstructorGenerator):
    def __init__(self, dataset: dict, root_path: str, version: int):
        super(DetectionDatasetConstructorGenerator, self).__init__(dataset, root_path, version, DetectionDatasetConstructor)
