from datasets.types.data_split import DataSplit
import os
import json
from typing import List, Dict, Tuple
from datasets.DET.constructor.base_interface import DetectionDatasetConstructor


def construct_COCO(constructor: DetectionDatasetConstructor, seed):
    root_path = seed.root_path
    include_crowd = seed.include_crowd
    data_split = seed.data_split
    version = seed.coco_version

    images: Dict[int: Tuple[str, List[Tuple[float, float, float, float]], List[int]]] = {}

    def _construct(image_folder_name: str, annotation_file_name: str):
        with open(os.path.join(root_path, 'annotations', annotation_file_name), 'r', encoding='utf-8') as fid:
            json_objects = json.load(fid)

        categories_ = json_objects['categories']
        constructor.set_category_id_name_map({category['id']: category['name'] for category in categories_})

        for annotation in json_objects['annotations']:
            if not include_crowd:
                if annotation['iscrowd'] != 0:
                    continue
            category_id = annotation['category_id']
            image_id = annotation['image_id']
            if image_id not in images:
                images[image_id] = ['', '', [], [], []]
            bbox = list(annotation['bbox'])
            images[image_id][2].append(bbox)
            images[image_id][3].append(category_id)
            if include_crowd:
                images[image_id][4].append({'iscrowd': annotation['iscrowd'] > 0})

        for image in json_objects['images']:
            image_id = image['id']
            if image_id not in images:
                continue
            file_name = image['file_name']
            images[image_id][0] = os.path.join('images', image_folder_name, file_name)
            images[image_id][1] = file_name
        constructor.set_total_number_of_images(len(images))
        for image_id, image_info in images.items():
            with constructor.new_image() as image_constructor:
                image_constructor.set_path(os.path.join(root_path, image_info[0]))
                if include_crowd:
                    for bounding_box, category_id, attributes in zip(image_info[2], image_info[3], image_info[4]):
                        with image_constructor.new_object() as object_constructor:
                            object_constructor.set_bounding_box(bounding_box)
                            object_constructor.set_category_id(category_id)
                            for attribute_key, attribute_value in attributes.items():
                                object_constructor.set_attribute(attribute_key, attribute_value)
                else:
                    for bounding_box, category_id in zip(image_info[2], image_info[3]):
                        with image_constructor.new_object() as object_constructor:
                            object_constructor.set_bounding_box(bounding_box)
                            object_constructor.set_category_id(category_id)
                image_constructor.set_attribute('image_id', image_id)

    if version == 2014:
        if data_split & DataSplit.Training:
            _construct('train2014', 'instances_train2014.json')
        if data_split & DataSplit.Validation:
            _construct('val2014', 'instances_val2014.json')
    elif version == 2017:
        if data_split & DataSplit.Training:
            _construct('train2017', 'instances_train2017.json')
        if data_split & DataSplit.Validation:
            _construct('val2017', 'instances_val2017.json')
    else:
        raise Exception
