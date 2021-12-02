from datasets.DET.constructor.base_interface import DetectionDatasetConstructor
import json
from data.types.bounding_box_format import BoundingBoxFormat
import os
from datasets.types.data_split import DataSplit


def _construct_Objects365(constructor: DetectionDatasetConstructor, images_path: str, annotation_file_path: str):
    with open(annotation_file_path, 'r', newline='\n') as f:
        annotation = json.load(f)

    constructor.set_category_id_name_map({category['id']: category['name'] for category in annotation['categories']})
    constructor.set_bounding_box_format(BoundingBoxFormat.XYWH)

    image_annotations = {}
    for image_attribute in annotation['images']:
        assert image_attribute['id'] not in image_annotations
        image_annotations[image_attribute['id']] = ((image_attribute['width'], image_attribute['height']), image_attribute['file_name'], [], [], [])

    for object_attribute in annotation['annotations']:
        image_annotation = image_annotations[object_attribute['image_id']]
        image_bboxes = image_annotation[2]
        image_category_ids = image_annotation[3]
        image_is_crowds = image_annotation[4]
        image_bboxes.append(object_attribute['bbox'])
        image_category_ids.append(object_attribute['category_id'])
        image_is_crowds.append(object_attribute['iscrowd'])

    constructor.set_total_number_of_images(len(image_annotations))

    for image_id, (image_size, image_file_name, image_bboxes, image_category_ids, image_is_crowds) in image_annotations.items():
        with constructor.new_image() as image_constructor:
            image_constructor.set_path(os.path.join(images_path, image_file_name), image_size)
            for object_bbox, object_category_id, object_is_crowd in zip(image_bboxes, image_category_ids, image_is_crowds):
                with image_constructor.new_object() as object_constructor:
                    object_constructor.set_category_id(object_category_id)
                    object_constructor.set_bounding_box(object_bbox)
                    object_constructor.set_attribute('is_crowd', object_is_crowd)


def construct_Objects365(constructor: DetectionDatasetConstructor, seed):
    root_path = seed.root_path
    data_split = seed.data_split

    if data_split & DataSplit.Training:
        _construct_Objects365(constructor, os.path.join(root_path, 'train'), os.path.join(root_path, 'objects365_train.json'))

    if data_split & DataSplit.Training:
        _construct_Objects365(constructor, os.path.join(root_path, 'val'), os.path.join(root_path, 'objects365_val.json'))
