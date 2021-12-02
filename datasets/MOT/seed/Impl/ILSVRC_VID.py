from datasets.MOT.constructor.base_interface import MultipleObjectTrackingDatasetConstructor
from datasets.types.data_split import DataSplit
import xml.etree.ElementTree as ET
import os
from data.types.bounding_box_format import BoundingBoxFormat


def construct_ILSVRC_VID(constructor: MultipleObjectTrackingDatasetConstructor, seed):
    root_path = seed.root_path
    data_type = seed.data_split
    wordnet_id_name_mapper = {
        'n02691156': 'airplane',
        'n02419796': 'antelope',
        'n02131653': 'bear',
        'n02834778': 'bicycle',
        'n01503061': 'bird',
        'n02924116': 'bus',
        'n02958343': 'car',
        'n02402425': 'cattle',
        'n02084071': 'dog',
        'n02121808': 'domestic cat',
        'n02503517': 'elephant',
        'n02118333': 'fox',
        'n02510455': 'giant panda',
        'n02342885': 'hamster',
        'n02374451': 'horse',
        'n02129165': 'lion',
        'n01674464': 'lizard',
        'n02484322': 'monkey',
        'n03790512': 'motorcycle',
        'n02324045': 'rabbit',
        'n02509815': 'red panda',
        'n02411705': 'sheep',
        'n01726692': 'snake',
        'n02355227': 'squirrel',
        'n02129604': 'tiger',
        'n04468005': 'train',
        'n01662784': 'turtle',
        'n04530566': 'watercraft',
        'n02062744': 'whale',
        'n02391049': 'zebra'
    }
    constructor.set_category_id_name_map({i: v for i, v in enumerate(wordnet_id_name_mapper.values())})
    wordnet_id_category_id_map = {k: i for i, k in enumerate(wordnet_id_name_mapper.keys())}

    image_path = os.path.join(root_path, 'Data', 'VID')
    annotation_path = os.path.join(root_path, 'Annotations', 'VID')

    sequence_names = []
    sequence_image_paths = []
    sequence_annotation_paths = []
    sequence_splits = []

    def _generate_and_append_sequence_paths(image_path, annotation_path, split):
        sequences = os.listdir(image_path)
        sequences.sort()
        for sequence in sequences:
            sequence_names.append(sequence)
            sequence_image_paths.append(os.path.join(image_path, sequence))
            sequence_annotation_paths.append(os.path.join(annotation_path, sequence))
            sequence_splits.append(split)

    if data_type & DataSplit.Training:
        train_image_path = os.path.join(image_path, 'train')
        train_annotation_path = os.path.join(annotation_path, 'train')
        image_paths = [os.path.join(train_image_path, data_folder_name) for data_folder_name in os.listdir(train_image_path)]
        annotation_paths = [os.path.join(train_annotation_path, data_folder_name) for data_folder_name in os.listdir(train_annotation_path)]
        image_paths.sort()
        annotation_paths.sort()
        for train_image_path, train_annotation_path in zip(image_paths, annotation_paths):
            _generate_and_append_sequence_paths(train_image_path, train_annotation_path, 'train')
    if data_type & DataSplit.Validation:
        _generate_and_append_sequence_paths(os.path.join(image_path, 'val'), os.path.join(annotation_path, 'val'), 'val')

    constructor.set_total_number_of_sequences(len(sequence_image_paths))
    constructor.set_bounding_box_format(BoundingBoxFormat.XYXY)

    for sequence, sequence_image_path, sequence_annotation_path, split in zip(sequence_names, sequence_image_paths, sequence_annotation_paths, sequence_splits):
        with constructor.new_sequence() as sequence_constructor:
            sequence_constructor.set_name(sequence)
            sequence_constructor.set_attribute('split', split)

            images = os.listdir(sequence_image_path)
            annotations = os.listdir(sequence_annotation_path)

            images = [image for image in images if image.endswith('.JPEG')]
            annotations = [annotation for annotation in annotations if annotation.endswith('.xml')]
            images.sort()
            annotations.sort()

            for image in images:
                with sequence_constructor.new_frame() as frame_constructor:
                    frame_constructor.set_path(os.path.join(sequence_image_path, image))

            object_ids = {}

            for index_of_frame in range(len(images)):
                annotation_file_path = os.path.join(sequence_annotation_path, annotations[index_of_frame])

                tree = ET.parse(annotation_file_path)

                root = tree.getroot()

                for object_child in root.findall('object'):
                    object_id = None
                    bounding_box = None
                    object_name = None
                    occluded = None
                    generated = None
                    for attribute in object_child:  # type: ET.Element
                        if attribute.tag == 'trackid':
                            assert object_id is None
                            object_id = int(attribute.text)
                        elif attribute.tag == 'name':
                            assert object_name is None
                            object_name = attribute.text
                        elif attribute.tag == 'bndbox':
                            assert bounding_box is None
                            xmin = None
                            xmax = None
                            ymin = None
                            ymax = None
                            for bounding_box_element in attribute:  # type: ET.Element
                                if bounding_box_element.tag == 'xmax':
                                    assert xmax is None
                                    xmax = int(bounding_box_element.text)
                                elif bounding_box_element.tag == 'xmin':
                                    assert xmin is None
                                    xmin = int(bounding_box_element.text)
                                elif bounding_box_element.tag == 'ymax':
                                    assert ymax is None
                                    ymax = int(bounding_box_element.text)
                                elif bounding_box_element.tag == 'ymin':
                                    assert ymin is None
                                    ymin = int(bounding_box_element.text)
                                else:
                                    raise Exception
                            bounding_box = [xmin, ymin, xmax, ymax]
                        elif attribute.tag == 'occluded':
                            occluded = int(attribute.text)
                        elif attribute.tag == 'generated':
                            generated = int(attribute.text)
                        else:
                            raise Exception
                    assert object_id is not None
                    assert bounding_box is not None
                    assert object_name is not None
                    assert occluded is not None
                    assert generated is not None

                    if object_id not in object_ids:
                        object_ids[object_id] = object_name
                        with sequence_constructor.new_object(object_id) as object_constructor:
                            object_constructor.set_category_id(wordnet_id_category_id_map[object_name])
                            object_constructor.set_attribute('WordNet ID', object_name)
                    else:
                        assert object_ids[object_id] == object_name

                    with sequence_constructor.open_frame(index_of_frame) as frame_constructor:
                        with frame_constructor.new_object(object_id) as object_constructor:
                            object_constructor.set_bounding_box(bounding_box)
                            object_constructor.merge_attributes({'occluded': occluded, 'generated': generated})
