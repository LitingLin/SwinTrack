import datasets.DET.dataset
from datasets.base.common.memory_mapped.constructor import memory_mapped_constructor_common_preliminary_works, \
    memory_mapped_constructor_get_bounding_box, memory_mapped_constructor_generate_bounding_box_matrix, \
    memory_mapped_constructor_generate_bounding_box_validity_flag_vector, memory_mapped_constructor_commit_data
from datasets.base.common.constructor import image_dataset_key_exclude_list, image_dataset_image_key_exclude_list, \
    image_dataset_object_key_exclude_list


def construct_detection_dataset_memory_mapped_from_base_image_dataset(base_dataset: dict, path: str):
    constructor, bounding_box_data_type = memory_mapped_constructor_common_preliminary_works(base_dataset, 'image',
                                                                                             path,
                                                                                             datasets.DET.dataset.__version__,
                                                                                             'Detection',
                                                                                             image_dataset_key_exclude_list)
    images_list = []

    for base_image in base_dataset['images']:
        object_attributes = []
        image_attributes = {
            'path': base_image['path'],
            'size': base_image['size'],
            'objects': object_attributes
        }
        if 'category_id' in base_image:
            image_attributes['category_id'] = base_image['category_id']

        bounding_box_matrix = []
        bounding_box_validity_flag_vector = []

        optional_image_attributes = {}
        optional_object_attributes = {}

        for base_image_key, base_image_value in base_image.items():
            if base_image_key in image_dataset_image_key_exclude_list:
                continue
            optional_image_attributes[base_image_key] = base_image_value

        if 'objects' in base_image:
            for index_of_base_object, base_object in enumerate(base_image['objects']):
                current_object_attributes = {}
                if 'category_id' in base_object:
                    current_object_attributes['category_id'] = base_object['category_id']

                current_optional_object_attributes = {}
                if 'bounding_box' in base_object:
                    object_bounding_box, object_bounding_box_validity = memory_mapped_constructor_get_bounding_box(
                        base_object)
                    bounding_box_matrix.append(object_bounding_box)
                    bounding_box_validity_flag_vector.append(object_bounding_box_validity)
                else:
                    bounding_box_matrix.append(None)
                    bounding_box_validity_flag_vector.append(None)

                for base_object_key, base_object_value in base_object.items():
                    if base_object_key in image_dataset_object_key_exclude_list:
                        continue
                    current_optional_object_attributes[base_object_key] = base_object_value

                object_attributes.append(current_object_attributes)
                if len(current_optional_object_attributes) > 0:
                    optional_object_attributes[index_of_base_object] = current_optional_object_attributes

        bounding_box_matrix, additional_bounding_box_validity_flag_vector = memory_mapped_constructor_generate_bounding_box_matrix(
            bounding_box_matrix, bounding_box_data_type)
        bounding_box_validity_flag_vector = memory_mapped_constructor_generate_bounding_box_validity_flag_vector(
            bounding_box_validity_flag_vector, additional_bounding_box_validity_flag_vector)

        if len(optional_object_attributes) > 0:
            optional_image_attributes['objects'] = optional_object_attributes

        if len(optional_image_attributes) == 0:
            optional_image_attributes = None

        images_list.append(
            (image_attributes, bounding_box_matrix, bounding_box_validity_flag_vector, optional_image_attributes))

    return memory_mapped_constructor_commit_data(images_list, constructor)
