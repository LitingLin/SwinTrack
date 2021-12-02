from datasets.types.data_split import DataSplit
import os
from datasets.SOT.constructor.base_interface import SingleObjectTrackingDatasetConstructor
from miscellanies.natural_keys import natural_keys
import numpy as np
from .data_specs.LaSOT_attr import LaSOT_attribute_names, LaSOT_attribute_values, LaSOT_attribute_short_names


_category_id_name_map = {0: 'airplane', 1: 'basketball', 2: 'bear', 3: 'bicycle', 4: 'bird', 5: 'boat', 6: 'book', 7: 'bottle', 8: 'bus', 9: 'car', 10: 'cat', 11: 'cattle', 12: 'chameleon', 13: 'coin', 14: 'crab', 15: 'crocodile', 16: 'cup', 17: 'deer', 18: 'dog', 19: 'drone', 20: 'electricfan', 21: 'elephant', 22: 'flag', 23: 'fox', 24: 'frog', 25: 'gametarget', 26: 'gecko', 27: 'giraffe', 28: 'goldfish', 29: 'gorilla', 30: 'guitar', 31: 'hand', 32: 'hat', 33: 'helmet', 34: 'hippo', 35: 'horse', 36: 'kangaroo', 37: 'kite', 38: 'leopard', 39: 'licenseplate', 40: 'lion', 41: 'lizard', 42: 'microphone', 43: 'monkey', 44: 'motorcycle', 45: 'mouse', 46: 'person', 47: 'pig', 48: 'pool', 49: 'rabbit', 50: 'racing', 51: 'robot', 52: 'rubicCube', 53: 'sepia', 54: 'shark', 55: 'sheep', 56: 'skateboard', 57: 'spider', 58: 'squirrel', 59: 'surfboard', 60: 'swing', 61: 'tank', 62: 'tiger', 63: 'train', 64: 'truck', 65: 'turtle', 66: 'umbrella', 67: 'volleyball', 68: 'yoyo', 69: 'zebra'}


def construct_LaSOT(constructor: SingleObjectTrackingDatasetConstructor, seed):
    root_path = seed.root_path
    data_type = seed.data_split

    subset_sequence_names = []

    if data_type & DataSplit.Training:
        with open(os.path.join(root_path, 'training_set.txt'), 'r') as f:
            content = f.readlines()
            content = [x.strip() for x in content]
            subset_sequence_names.extend(content)

    if data_type & DataSplit.Validation:
        with open(os.path.join(root_path, 'testing_set.txt'), 'r') as f:
            content = f.readlines()
            content = [x.strip() for x in content]
            subset_sequence_names.extend(content)

    class_names = os.listdir(root_path)
    class_names = [class_name for class_name in class_names if os.path.isdir(os.path.join(root_path, class_name))]
    class_names.sort()

    constructor.set_category_id_name_map(_category_id_name_map)
    category_name_id_map = {v: k for k, v in _category_id_name_map.items()}

    constructor.set_attribute('sequence attribute full names', LaSOT_attribute_names)
    constructor.set_attribute('sequence attribute names', LaSOT_attribute_short_names)

    tasks = []

    for class_name in class_names:
        if not any(sequenceName.startswith(class_name) for sequenceName in subset_sequence_names):
            raise Exception

        class_path = os.path.join(root_path, class_name)
        sequence_names = os.listdir(class_path)
        sequence_names = [sequence_name for sequence_name in sequence_names if
                          os.path.isdir(os.path.join(class_path, sequence_name))]
        sequence_names.sort(key=natural_keys)

        for sequence_name in sequence_names:
            if sequence_name not in subset_sequence_names:
                continue
            tasks.append((class_name, sequence_name))

    constructor.set_total_number_of_sequences(len(tasks))

    for class_name, sequence_name in tasks:
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
            bounding_boxes[:, 0:2] -= 1
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

            for index_of_frame, (bounding_box, is_fully_occlusion, is_out_of_view) in enumerate(zip(bounding_boxes, is_fully_occlusions, is_out_of_views)):
                with sequence_constructor.open_frame(index_of_frame) as frame_constructor:
                    frame_constructor.set_bounding_box(bounding_box.tolist(), validity=not (is_fully_occlusion or is_out_of_view))
                    frame_constructor.set_object_attribute('occlusion', is_fully_occlusion.item())
                    frame_constructor.set_object_attribute('out of view', is_out_of_view.item())
