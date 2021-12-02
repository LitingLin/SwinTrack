import datasets.MOT.dataset
import numpy as np
from datasets.base.common.memory_mapped.constructor import memory_mapped_constructor_common_preliminary_works, \
    memory_mapped_constructor_commit_data, memory_mapped_constructor_get_bounding_box
from datasets.base.common.constructor import video_dataset_key_exclude_list, video_dataset_sequence_key_exclude_list, \
    video_dataset_frame_key_exclude_list, video_dataset_sequence_object_key_exclude_list, \
    video_dataset_frame_object_key_exclude_list


def construct_multiple_object_tracking_dataset_memory_mapped_from_base_video_dataset(base_dataset: dict, path: str):
    constructor, bounding_box_data_type = memory_mapped_constructor_common_preliminary_works(base_dataset, 'video',
                                                                                             path,
                                                                                             datasets.MOT.dataset.__version__,
                                                                                             'MultipleObjectTracking',
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

        # sequence_frame_object_attribute_indices_vector = [] # a) index the objects in a frame in object* matrix b) being indexed by 'object_attributes_index_range' in frame_attribute
        sequence_frame_object_id_vector = []  # a) object ids in a frame b) being indexed by 'object_attributes_index_range' in frame_attribute
        sequence_object_frame_index_vector = []  # a) frame indices the object occurs b) being indexed by 'object_attributes_index_range' in object_attribute sequential c) being indexed by sequence_frame_object_attribute_indices_vector unsequentail
        sequence_object_bounding_box_matrix = []
        sequence_object_bounding_box_validity_flag_matrix = []

        sequence_object_frame_index_vector_dict = {}
        sequence_object_bounding_box_matrix_dict = {}
        sequence_object_bounding_box_validity_flag_matrix_dict = {}

        sequence_frame_sizes = []
        first_frame_size = base_sequence['frames'][0]['size']
        sequence_frame_size_all_equal = True

        sequence_has_bounding_box_annotation = False

        current_annotation_index = 0

        additional_sequence_attributes_dict = {}
        for base_sequence_key, base_sequence_value in base_sequence.items():
            if base_sequence_key in video_dataset_sequence_key_exclude_list:
                continue
            additional_sequence_attributes_dict[base_sequence_key] = base_sequence_value

        sequence_attributes['objects'] = {}
        if 'objects' in base_sequence:
            for base_sequence_object in base_sequence['objects']:
                object_id = base_sequence_object['id']
                sequence_object_attribute_dict = {}
                sequence_attributes['objects'][object_id] = sequence_object_attribute_dict
                if 'category_id' in base_sequence_object:
                    sequence_object_attribute_dict['category_id'] = base_sequence_object['category_id']
                sequence_object_frame_index_vector_dict[object_id] = []
                sequence_object_bounding_box_matrix_dict[object_id] = []
                sequence_object_bounding_box_validity_flag_matrix_dict[object_id] = []
                additional_sequence_object_attributes_dict = {}
                for base_sequence_object_key, base_sequence_object_value in base_sequence_object.items():
                    if base_sequence_object_key in video_dataset_sequence_object_key_exclude_list:
                        continue
                    additional_sequence_object_attributes_dict[base_sequence_object_key] = base_sequence_object_value
                if len(additional_sequence_object_attributes_dict) > 0:
                    if 'objects' not in additional_sequence_attributes_dict:
                        additional_sequence_attributes_dict['objects'] = {}
                    additional_sequence_attributes_dict['objects'][
                        object_id] = additional_sequence_object_attributes_dict

        sequence_attributes['frames'] = []

        for index_of_base_frame, base_frame in enumerate(base_sequence['frames']):
            additional_frame_attributes_dict = {}

            for base_frame_key, base_frame_value in base_frame.items():
                if base_frame_key in video_dataset_frame_key_exclude_list:
                    continue
                additional_frame_attributes_dict[base_frame_key] = base_frame_value

            if first_frame_size != base_frame['size']:
                sequence_frame_size_all_equal = False
            sequence_frame_sizes.append(base_frame['size'])
            frame_annotation_begin_index = current_annotation_index

            if 'objects' in base_frame:
                for base_frame_object in base_frame['objects']:
                    if 'id' not in base_frame_object:
                        continue
                    object_id = base_frame_object['id']
                    if object_id not in sequence_object_frame_index_vector_dict:
                        continue

                    assert index_of_base_frame not in sequence_object_frame_index_vector_dict[object_id]
                    sequence_object_frame_index_vector_dict[object_id].append(index_of_base_frame)
                    if 'bounding_box' in base_frame_object:
                        bounding_box, bounding_box_validity_flag = memory_mapped_constructor_get_bounding_box(
                            base_frame_object)
                        sequence_object_bounding_box_matrix_dict[object_id].append(bounding_box)
                        sequence_object_bounding_box_validity_flag_matrix_dict[object_id].append(
                            bounding_box_validity_flag)
                        sequence_has_bounding_box_annotation = True
                    else:
                        sequence_object_bounding_box_matrix_dict[object_id].append([-1, -1, -1, -1])
                        sequence_object_bounding_box_validity_flag_matrix_dict[object_id].append(False)

                    sequence_frame_object_id_vector.append(object_id)
                    current_annotation_index += 1

                    additional_frame_object_attributes_dict = {}
                    for base_frame_object_key, base_frame_object_value in base_frame_object.items():
                        if base_frame_object_key in video_dataset_frame_object_key_exclude_list:
                            continue
                        additional_frame_object_attributes_dict[base_frame_object_key] = base_frame_object_value
                    if len(additional_frame_object_attributes_dict) > 0:
                        if 'objects' not in additional_frame_attributes_dict:
                            additional_frame_attributes_dict['objects'] = {}
                        additional_frame_attributes_dict['objects'][object_id] = additional_frame_object_attributes_dict
            frame_object_attributes_index_range = (frame_annotation_begin_index, current_annotation_index)
            frame_attribute = {
                'path': base_frame['path'],
                'object_attributes_index_range': frame_object_attributes_index_range
            }
            sequence_attributes['frames'].append(frame_attribute)
            if len(additional_frame_attributes_dict) > 0:
                if 'frames' not in additional_sequence_attributes_dict:
                    additional_sequence_attributes_dict['frames'] = {}
                additional_sequence_attributes_dict['frames'][index_of_base_frame] = additional_frame_attributes_dict

        sequence_frame_object_id_vector = np.array(sequence_frame_object_id_vector)
        sequence_frame_object_attribute_indices_vector = np.zeros(shape=sequence_frame_object_id_vector.shape,
                                                                  dtype=np.int32)
        current_annotation_index = 0
        for object_id in sequence_object_frame_index_vector_dict.keys():
            object_annotation_length = len(sequence_object_frame_index_vector_dict[object_id])
            sequence_object_attribute_dict = sequence_attributes['objects'][object_id]
            sequence_object_frame_index_vector.extend(sequence_object_frame_index_vector_dict[object_id])
            sequence_object_bounding_box_matrix.extend(sequence_object_bounding_box_matrix_dict[object_id])
            sequence_object_bounding_box_validity_flag_matrix.extend(
                sequence_object_bounding_box_validity_flag_matrix_dict[object_id])
            for i, frame_index in enumerate(sequence_object_frame_index_vector_dict[object_id]):
                frame_object_attributes_index_range = sequence_attributes['frames'][frame_index][
                    'object_attributes_index_range']
                object_ids = sequence_frame_object_id_vector[
                             frame_object_attributes_index_range[0]: frame_object_attributes_index_range[1]]
                index = np.where(object_ids == object_id)[0][0]
                sequence_frame_object_attribute_indices_vector[
                    frame_object_attributes_index_range[0] + index] = current_annotation_index + i
            sequence_object_attribute_dict['object_attributes_index_range'] = (
            current_annotation_index, current_annotation_index + object_annotation_length)
            current_annotation_index += object_annotation_length

        if sequence_frame_size_all_equal:
            sequence_attributes['frame_size'] = first_frame_size
            sequence_image_size_matrix = None
        else:
            sequence_image_size_matrix = np.array(sequence_frame_sizes)

        if sequence_frame_object_id_vector.shape[0] == 0:
            sequence_frame_object_id_vector = None
        if sequence_frame_object_attribute_indices_vector.shape[0] == 0:
            sequence_frame_object_attribute_indices_vector = None
        if len(sequence_object_frame_index_vector) > 0:
            sequence_object_frame_index_vector = np.array(sequence_object_frame_index_vector)
        else:
            sequence_object_frame_index_vector = None

        if sequence_has_bounding_box_annotation:
            sequence_object_bounding_box_matrix = np.array(sequence_object_bounding_box_matrix, dtype=bounding_box_data_type)
            if all([flag is None for flag in sequence_object_bounding_box_validity_flag_matrix]):
                sequence_object_bounding_box_validity_flag_matrix = None
            else:
                sequence_object_bounding_box_validity_flag_matrix = [False if flag is None else flag for flag in sequence_object_bounding_box_validity_flag_matrix]
                sequence_object_bounding_box_validity_flag_matrix = np.array(
                    sequence_object_bounding_box_validity_flag_matrix)
        else:
            sequence_object_bounding_box_matrix = None
            sequence_object_bounding_box_validity_flag_matrix = None

        if len(additional_sequence_attributes_dict) == 0:
            additional_sequence_attributes_dict = None

        sequences_list.append((sequence_attributes, sequence_image_size_matrix,
                               sequence_frame_object_attribute_indices_vector, sequence_frame_object_id_vector,
                               sequence_object_frame_index_vector, sequence_object_bounding_box_matrix,
                               sequence_object_bounding_box_validity_flag_matrix,
                               additional_sequence_attributes_dict))

    return memory_mapped_constructor_commit_data(sequences_list, constructor)
