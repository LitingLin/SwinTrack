from datasets.types.data_split import DataSplit
import os
from datasets.base.meta.imagenet_200 import ImageNet200
from datasets.DET.constructor.base_interface import DetectionDatasetConstructor
from data.types.bounding_box_format import BoundingBoxFormat


def construct_ILSVRC_DET(constructor: DetectionDatasetConstructor, seed):
    imagenet_id_index_map = {id_: index for index, id_ in enumerate(ImageNet200.wn_id)}
    root_path = seed.root_path
    data_split = seed.data_split

    constructor.set_category_id_name_map({index: name for index, name in enumerate(ImageNet200.names)})

    annotation_path = os.path.join(root_path, 'Annotations', 'DET')
    annotation_paths = []
    image_paths = []

    if data_split & DataSplit.Training:
        path = os.path.join(annotation_path, 'train')
        for dirpath, _, filelist in os.walk(path):
            if len(filelist) > 0:
                annotation_paths.append(dirpath)
                image_paths.append(dirpath.replace('Annotations', 'Data', 1))
    elif data_split & DataSplit.Validation:
        annotation_paths.append(os.path.join('Annotations', 'DET', 'val'))
        image_paths.append(os.path.join('Data', 'DET', 'val'))

    annotation_file_paths = []
    image_file_paths = []

    for annotation_path, image_path in zip(annotation_paths, image_paths):
        annotation_file_names = os.listdir(os.path.join(root_path, annotation_path))
        for annotation_file_name in annotation_file_names:
            annotation_file = os.path.join(root_path, annotation_path, annotation_file_name)
            image_name = annotation_file_name[:annotation_file_name.rfind('.')]
            current_image = os.path.join(root_path, image_path, image_name + '.JPEG')
            annotation_file_paths.append(annotation_file)
            image_file_paths.append(current_image)

    constructor.set_total_number_of_images(len(annotation_file_paths))
    constructor.set_bounding_box_format(BoundingBoxFormat.XYXY)

    for annotation_file_path, image_file_path in zip(annotation_file_paths, image_file_paths):
        with open(annotation_file_path, 'r') as fid:
            file_content = fid.read()

        offset = 0
        bounding_boxes = []
        object_category_indices = []

        def _findNextObject(file_content: str, begin_index: int):
            object_name_begin_index = file_content.find('<name>', begin_index)
            if object_name_begin_index == -1:
                return None
            object_name_end_index = file_content.find('</name>', object_name_begin_index)
            xmin_begin_index = file_content.find('<xmin>', object_name_end_index)
            xmin_end_index = file_content.find('</xmin>', xmin_begin_index)
            xmax_begin_index = file_content.find('<xmax>', xmin_end_index)
            xmax_end_index = file_content.find('</xmax>', xmax_begin_index)
            ymin_begin_index = file_content.find('<ymin>', xmax_end_index)
            ymin_end_index = file_content.find('</ymin>', ymin_begin_index)
            ymax_begin_index = file_content.find('<ymax>', ymin_end_index)
            ymax_end_index = file_content.find('</ymax>', ymax_begin_index)

            object_name = file_content[object_name_begin_index + len('<name>'): object_name_end_index]
            object_category_index = imagenet_id_index_map[object_name]

            xmax = int(file_content[xmax_begin_index + 6: xmax_end_index])
            xmin = int(file_content[xmin_begin_index + 6: xmin_end_index])
            ymax = int(file_content[ymax_begin_index + 6: ymax_end_index])
            ymin = int(file_content[ymin_begin_index + 6: ymin_end_index])

            return object_category_index, xmin, ymin, xmax, ymax, ymax_end_index + 7

        while True:
            next_object_in_annotation_file = _findNextObject(file_content, offset)
            if next_object_in_annotation_file is None:
                break
            object_category_index, xmin, ymin, xmax, ymax, offset = next_object_in_annotation_file
            bounding_boxes.append((xmin, ymin, xmax, ymax))
            object_category_indices.append(object_category_index)
        if len(bounding_boxes) > 0:
            with constructor.new_image() as image_constructor:
                image_constructor.set_path(image_file_path)
                for bounding_box, object_category_index in zip(bounding_boxes, object_category_indices):
                    with image_constructor.new_object() as object_constructor:
                        object_constructor.set_bounding_box(bounding_box)
                        object_constructor.set_category_id(object_category_index)
