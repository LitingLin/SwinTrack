import datasets.SOT.dataset
import numpy as np

from datasets.base.common.memory_mapped.constructor import memory_mapped_constructor_common_preliminary_works, \
    memory_mapped_constructor_commit_data, memory_mapped_constructor_get_bounding_box, \
    memory_mapped_constructor_generate_bounding_box_matrix, \
    memory_mapped_constructor_generate_bounding_box_validity_flag_vector
from datasets.base.common.constructor import video_dataset_key_exclude_list, video_dataset_sequence_key_exclude_list, \
    video_dataset_frame_key_exclude_list, video_dataset_sequence_object_key_exclude_list, \
    video_dataset_frame_object_key_exclude_list


def construct_single_object_tracking_dataset_memory_mapped_from_base_video_dataset(base_dataset: dict, path: str):
    constructor, dataset_bounding_box_data_type = memory_mapped_constructor_common_preliminary_works(base_dataset,
                                                                                                     'video', path,
                                                                                                     datasets.SOT.dataset.__version__,
                                                                                                     'SingleObjectTracking',
                                                                                                     video_dataset_key_exclude_list)

    sequences_list = []

    for base_sequence in base_dataset['sequences']:
        if len(base_sequence['frames']) == 0:
            continue

        sequence_attributes = {
            'name': base_sequence['name'],
            'path': base_sequence['path']
        }
        if 'fps' in base_sequence:
            sequence_attributes['fps'] = base_sequence['fps']

        optional_sequence_attributes = {}
        optional_frame_attributes = {}

        sequence_object_id = None
        sequence_category_id = None

        optional_sequence_object_attributes = {}

        for base_sequence_key, base_sequence_value in base_sequence.items():
            if base_sequence_key in video_dataset_sequence_key_exclude_list:
                continue
            optional_sequence_attributes[base_sequence_key] = base_sequence_value

        if 'objects' in base_sequence:
            assert len(base_sequence['objects']) == 1
            base_sequence_object = base_sequence['objects'][0]
            sequence_object_id = base_sequence_object['id']
            if 'category_id' in base_sequence_object:
                sequence_category_id = base_sequence_object['category_id']
            for base_sequence_object_key, base_sequence_object_value in base_sequence_object.items():
                if base_sequence_object_key in video_dataset_sequence_object_key_exclude_list:
                    continue
                optional_sequence_object_attributes[base_sequence_object_key] = base_sequence_object_value
        if sequence_category_id is not None:
            sequence_attributes['category_id'] = sequence_category_id
        if len(optional_sequence_object_attributes) > 0:
            optional_sequence_attributes['object'] = optional_sequence_object_attributes

        frame_attributes_list = []

        sequence_bounding_box_matrix = []
        sequence_bounding_box_validity_flag_vector = []

        sequence_frame_sizes = []
        first_frame_size = base_sequence['frames'][0]['size']
        sequence_frame_size_all_equal = True

        for index_of_base_frame, base_frame in enumerate(base_sequence['frames']):
            current_optional_frame_attributes = {}
            for base_frame_key, base_frame_value in base_frame.items():
                if base_frame_key in video_dataset_frame_key_exclude_list:
                    continue
                current_optional_frame_attributes[base_frame_key] = base_frame_value

            frame_attribute = {
                'path': base_frame['path']
            }

            if first_frame_size != base_frame['size']:
                sequence_frame_size_all_equal = False
            sequence_frame_sizes.append(base_frame['size'])
            base_object = None
            if 'objects' in base_frame:
                if len(base_frame['objects']) > 0:
                    number_of_same_objects_id = 0
                    sequence_object_in_current_frame = None
                    for object_ in base_frame['objects']:
                        if 'id' in object_ and object_['id'] == sequence_object_id:
                            number_of_same_objects_id += 1
                            sequence_object_in_current_frame = object_
                    assert number_of_same_objects_id < 2
                    if number_of_same_objects_id == 1:
                        base_object = sequence_object_in_current_frame

            frame_bounding_box = None
            frame_bounding_box_validity = None

            if base_object is not None:
                current_frame_optional_object_attributes = {}
                for base_object_key, base_object_value in base_object.items():
                    if base_object_key in video_dataset_frame_object_key_exclude_list:
                        continue
                    current_frame_optional_object_attributes[base_object_key] = base_object_value
                if len(current_frame_optional_object_attributes) > 0:
                    optional_frame_attributes['object'] = current_frame_optional_object_attributes

                if 'bounding_box' in base_object:
                    frame_bounding_box, frame_bounding_box_validity = memory_mapped_constructor_get_bounding_box(
                        base_object)

            sequence_bounding_box_matrix.append(frame_bounding_box)
            sequence_bounding_box_validity_flag_vector.append(frame_bounding_box_validity)
            frame_attributes_list.append(frame_attribute)
            if len(current_optional_frame_attributes) > 0:
                optional_frame_attributes[index_of_base_frame] = current_optional_frame_attributes

        if sequence_frame_size_all_equal:
            sequence_attributes['frame_size'] = first_frame_size
            sequence_frame_sizes = None
        else:
            sequence_frame_sizes = np.array(sequence_frame_sizes)
        sequence_attributes['frames'] = frame_attributes_list

        sequence_bounding_box_matrix, additional_sequence_bounding_box_validity_flag_vector = memory_mapped_constructor_generate_bounding_box_matrix(
            sequence_bounding_box_matrix, dataset_bounding_box_data_type)
        sequence_bounding_box_validity_flag_vector = memory_mapped_constructor_generate_bounding_box_validity_flag_vector(
            sequence_bounding_box_validity_flag_vector, additional_sequence_bounding_box_validity_flag_vector)

        if len(optional_frame_attributes) > 0:
            optional_sequence_attributes['frames'] = optional_frame_attributes
        if len(optional_sequence_attributes) == 0:
            optional_sequence_attributes = None

        sequences_list.append((sequence_attributes, sequence_frame_sizes, sequence_bounding_box_matrix,
                               sequence_bounding_box_validity_flag_vector, optional_sequence_attributes))

    return memory_mapped_constructor_commit_data(sequences_list, constructor)
