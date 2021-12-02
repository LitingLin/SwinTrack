from datasets.SOT.constructor.base_interface import SingleObjectTrackingDatasetConstructor
import os
from datasets.types.data_split import DataSplit
from miscellanies.natural_keys import natural_keys
import numpy as np
from .data_specs.LaSOT_attr import LaSOT_attribute_names, LaSOT_attribute_values, LaSOT_attribute_short_names


_category_id_name_map = {0: 'atv', 1: 'badminton', 2: 'cosplay', 3: 'dancingshoe', 4: 'footbag', 5: 'frisbee', 6: 'jianzi', 7: 'lantern', 8: 'misc', 9: 'opossum', 10: 'paddle', 11: 'raccoon', 12: 'rhino', 13: 'skatingshoe', 14: 'wingsuit'}


def construct_LaSOT_Extension(constructor: SingleObjectTrackingDatasetConstructor, seed):
    assert seed.data_split == DataSplit.Full
    root_path = seed.root_path

    class_names = os.listdir(root_path)
    class_names = [class_name for class_name in class_names if os.path.isdir(os.path.join(root_path, class_name))]
    class_names.sort()
    sequences = []
    for class_name in class_names:
        class_path = os.path.join(root_path, class_name)
        sequence_names = os.listdir(class_path)
        sequence_names = [sequence_name for sequence_name in sequence_names if
                          os.path.isdir(os.path.join(class_path, sequence_name))]
        sequence_names.sort(key=natural_keys)

        for sequence_name in sequence_names:
            sequences.append((class_name, sequence_name))

    constructor.set_category_id_name_map(_category_id_name_map)
    category_name_id_map = {v: k for k, v in _category_id_name_map.items()}

    constructor.set_attribute('sequence attribute full names', LaSOT_attribute_names)
    constructor.set_attribute('sequence attribute names', LaSOT_attribute_short_names)

    constructor.set_total_number_of_sequences(len(sequences))

    for class_name, sequence_name in sequences:
        category_id = category_name_id_map[class_name]
        class_path = os.path.join(root_path, class_name)
        with constructor.new_sequence(category_id) as sequence_constructor:
            sequence_constructor.set_name(sequence_name)

            sequence_attribute = LaSOT_attribute_values[sequence_name]
            for sequence_attribute_name, attribute_value in zip(LaSOT_attribute_short_names, sequence_attribute):
                sequence_constructor.set_attribute(sequence_attribute_name, attribute_value)

            sequence_path = os.path.join(class_path, sequence_name)
            groundtruth_file_path = os.path.join(sequence_path, 'groundtruth.txt')
            bounding_boxes = np.loadtxt(groundtruth_file_path, dtype=np.int, delimiter=',')
            full_occlusion_file_path = os.path.join(sequence_path, 'full_occlusion.txt')
            is_fully_occlusions = np.loadtxt(full_occlusion_file_path, dtype=np.bool, delimiter=',')
            out_of_view_file_path = os.path.join(sequence_path, 'out_of_view.txt')
            is_out_of_views = np.loadtxt(out_of_view_file_path, dtype=np.bool, delimiter=',')
            images_path = os.path.join(sequence_path, 'img')
            if len(bounding_boxes) != len(is_fully_occlusions) or len(is_fully_occlusions) != len(is_out_of_views):
                raise Exception('annotation length mismatch in {}'.format(sequence_path))

            images = os.listdir(images_path)
            images = [image for image in images if image.endswith('.jpg')]
            images.sort()
            for image in images:
                image_path = os.path.join(images_path, image)
                with sequence_constructor.new_frame() as frame_constructor:
                    frame_constructor.set_path(image_path)

            for index_of_frame, (bounding_box, is_fully_occlusion, is_out_of_view) in enumerate(
                    zip(bounding_boxes, is_fully_occlusions, is_out_of_views)):
                with sequence_constructor.open_frame(index_of_frame) as frame_constructor:
                    frame_constructor.set_bounding_box(bounding_box.tolist(), validity=not(is_fully_occlusion or is_out_of_view))
                    frame_constructor.set_object_attribute('occlusion', is_fully_occlusion.item())
                    frame_constructor.set_object_attribute('out of view', is_out_of_view.item())
