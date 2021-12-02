import os
from datasets.types.data_split import DataSplit
from datasets.SOT.constructor.base_interface import SingleObjectTrackingDatasetConstructor
import numpy as np


def construct_TrackingNet(constructor: SingleObjectTrackingDatasetConstructor, seed):
    root_path = seed.root_path
    data_type = seed.data_split
    enable_set_ids = seed.enable_set_ids
    sequence_name_class_map_file_path = seed.sequence_name_class_map_file_path
    if data_type != DataSplit.Training and enable_set_ids is not None:
        raise Exception("unsupported configuration")

    sequence_name_class_map = {}

    if sequence_name_class_map_file_path is None:
        sequence_name_class_map_file_path = os.path.join(os.path.dirname(__file__), 'data_specs', 'trackingnet_sequence_classes_map.txt')

    for line in open(sequence_name_class_map_file_path, 'r', encoding='utf-8'):
        line = line.strip()
        name, category = line.split('\t')
        sequence_name_class_map[name] = category

    categories = set(sequence_name_class_map.values())
    category_id_name_map = {i: v for i, v in enumerate(categories)}
    category_name_id_map = {v: i for i, v in enumerate(categories)}

    if enable_set_ids is not None:
        trackingNetSubsets = ['TRAIN_{}'.format(v) for v in enable_set_ids]
    else:
        trackingNetSubsets = []
        if data_type & DataSplit.Training:
            trackingNetSubsets = ['TRAIN_{}'.format(v) for v in range(12)]
        if data_type & DataSplit.Testing:
            trackingNetSubsets.append('TEST')

    sequence_list = []

    for subset in trackingNetSubsets:
        subset_path = os.path.join(root_path, subset)
        frames_path = os.path.join(subset_path, 'frames')
        anno_path = os.path.join(subset_path, 'anno')

        bounding_box_annotation_files = os.listdir(anno_path)
        bounding_box_annotation_files = [bounding_box_annotation_file for bounding_box_annotation_file in
                                         bounding_box_annotation_files if bounding_box_annotation_file.endswith('.txt')]
        bounding_box_annotation_files.sort()

        sequences = [sequence[:-4] for sequence in bounding_box_annotation_files]
        for sequence, bounding_box_annotation_file in zip(sequences, bounding_box_annotation_files):
            sequence_image_path = os.path.join(frames_path, sequence)
            bounding_box_annotation_file_path = os.path.join(anno_path, bounding_box_annotation_file)
            sequence_list.append((sequence, sequence_image_path, bounding_box_annotation_file_path))

    constructor.set_category_id_name_map(category_id_name_map)
    constructor.set_total_number_of_sequences(len(sequence_list))

    for sequence, sequence_image_path, sequence_bounding_box_annotation_file_path in sequence_list:
        with constructor.new_sequence(category_name_id_map[sequence_name_class_map[sequence]]) as sequence_constructor:
            sequence_constructor.set_name(sequence)
            bounding_boxes = np.loadtxt(sequence_bounding_box_annotation_file_path, dtype=np.float, delimiter=',')
            images = os.listdir(sequence_image_path)
            images = [image for image in images if image.endswith('.jpg')]
            if bounding_boxes.ndim == 2:
                is_testing_sequence = False
                assert len(images) == len(bounding_boxes)
            else:
                is_testing_sequence = True
                assert bounding_boxes.ndim == 1 and bounding_boxes.shape[0] == 4

            for i in range(len(images)):
                image_file_name = '{}.jpg'.format(i)
                image_file_path = os.path.join(sequence_image_path, image_file_name)
                with sequence_constructor.new_frame() as frame_constructor:
                    frame_constructor.set_path(image_file_path)
                    if is_testing_sequence:
                        if i == 0:
                            frame_constructor.set_bounding_box(bounding_boxes.tolist())
                    else:
                        frame_constructor.set_bounding_box(bounding_boxes[i].tolist())
