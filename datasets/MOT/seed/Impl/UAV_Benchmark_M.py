import os
from datasets.MOT.constructor.base_interface import MultipleObjectTrackingDatasetConstructor
from datasets.types.data_split import DataSplit


def construct_UAV_Benchmark_M(constructor: MultipleObjectTrackingDatasetConstructor, seed):
    root_path = seed.root_path
    annotation_path = seed.annotation_path
    data_type = seed.data_split

    if annotation_path is None:
        annotation_path = os.path.join(root_path, 'UAV-benchmark-MOTD_v1.0')

    annotation_path = os.path.join(annotation_path, 'GT')
    annotation_files = os.listdir(annotation_path)
    annotation_files = [annotation_file for annotation_file in annotation_files if annotation_file.endswith('_gt_whole.txt')]
    annotation_files.sort()

    def _read_sequence_attrs(root_path: str):
        attr_path = os.path.join(root_path, 'M_attr')
        train_attr_path = os.path.join(attr_path, 'train')
        val_attr_path = os.path.join(attr_path, 'test')

        def _read_attrs(path: str):
            attrs = {}
            files = os.listdir(path)
            files = [file for file in files if file.endswith('_attr.txt')]
            files.sort()
            for file in files:
                attr_file_path = os.path.join(path, file)
                sequence_name = file[:-9]
                # daylight, night, fog; low-alt, medium-alt, high-alt; front-view, side-view,	bird-view; long-term.
                with open(attr_file_path) as fid:
                    file_content = fid.read()
                file_content = file_content.strip()
                values = file_content.split(',')
                assert len(values) == 10
                attrs[sequence_name] = {
                    'daylight': bool(int(values[0])), 'night': bool(int(values[1])), 'fog': bool(int(values[2])),
                    'low-alt': bool(int(values[3])), 'medium-alt': bool(int(values[4])), 'high-alt': bool(int(values[5])),
                    'front-view': bool(int(values[6])), 'side-view': bool(int(values[7])), 'bird-view': bool(int(values[8])),
                    'long-term': bool(int(values[9]))
                }
            return attrs
        return _read_attrs(train_attr_path), _read_attrs(val_attr_path)

    train_sequence_attrs, val_sequence_attrs = _read_sequence_attrs(root_path)

    constructor.set_category_id_name_map({0: 'car', 1: 'truck', 2: 'bus'})
    constructor.set_total_number_of_sequences(len(annotation_files))

    for annotation_file in annotation_files:
        annotation_file_path = os.path.join(annotation_path, annotation_file)
        sequence_name = annotation_file[:-len('_gt_whole.txt')]

        sequence_attrs = None
        if data_type & DataSplit.Training and sequence_name in train_sequence_attrs:
            sequence_attrs = train_sequence_attrs[sequence_name]
        if data_type & DataSplit.Validation and sequence_name in val_sequence_attrs:
            sequence_attrs = val_sequence_attrs[sequence_name]

        if sequence_attrs is None:
            continue

        with constructor.new_sequence() as sequence_constructor:
            sequence_constructor.set_name(sequence_name)
            sequence_constructor.merge_attributes(sequence_attrs)

            sequence_images_path = os.path.join(root_path, sequence_name)
            images = os.listdir(sequence_images_path)
            images = [image for image in images if image.endswith('.jpg')]
            images.sort()
            for image in images:
                with sequence_constructor.new_frame() as frame_constructor:
                    frame_constructor.set_path(os.path.join(sequence_images_path, image))

            object_categories = {}
            for line in open(annotation_file_path, 'r', encoding='utf-8'):
                line = line.strip()
                if len(line) == 0:
                    continue
                '''
                <frame_index>,<target_id>,<bbox_left>,<bbox_top>,<bbox_width>,<bbox_height>,<out-of-view>,<occlusion>,<object_category>
    
            -----------------------------------------------------------------------------------------------------------------------------------
                   Name	                                      Description
            -----------------------------------------------------------------------------------------------------------------------------------
                <frame_index>   The frame index of the video frame
    
                <target_id>	  In the GROUNDTRUTH file, the identity of the target is used to provide the temporal corresponding 
                              relation of the bounding boxes in different frames.
    
                <bbox_left>	          The x coordinate of the top-left corner of the predicted bounding box
    
                <bbox_top>	          The y coordinate of the top-left corner of the predicted object bounding box
    
                <bbox_width>	  The width in pixels of the predicted object bounding box
    
                <bbox_height>	  The height in pixels of the predicted object bounding box
    
                <out-of-view>	     The score in the GROUNDTRUTH file indicates the degree of object parts appears outside a frame 
                              (i.e., 'no-out'= 1,'medium-out' =2,'small-out'=3).
    
                <occlusion>	  The score in the GROUNDTRUTH fileindicates the fraction of objects being occluded.
                                (i.e.,'no-occ'=1,'lagre-occ'=2,'medium-occ'=3,'small-occ'=4).
    
                <object_category>	  The object category indicates the type of annotated object, (i.e.,car(1), truck(2), bus(3))
            '''
                values = line.split(',')
                assert len(values) == 9
                frame_index = int(values[0]) - 1
                target_id = int(values[1])
                bounding_box = [int(values[2]), int(values[3]), int(values[4]), int(values[5])]
                out_of_view = int(values[6])
                occlusion = int(values[7])
                object_category = int(values[8])

                if target_id not in object_categories:
                    object_categories[target_id] = [object_category]
                else:
                    object_categories[target_id].append(object_category)
                is_present = not(out_of_view == 2 or occlusion == 2)
                with sequence_constructor.open_frame(frame_index) as frame_constructor:
                    with frame_constructor.new_object(target_id) as object_constructor:
                        object_constructor.set_bounding_box(bounding_box, validity=is_present)
                        object_constructor.merge_attributes({'out_of_view': out_of_view, 'occlusion': occlusion})
            for object_id, object_category in object_categories.items():
                count = [object_category.count(1), object_category.count(2), object_category.count(3)]
                index = count.index(max(count))
                with sequence_constructor.new_object(target_id) as object_constructor:
                    object_constructor.set_category_id(index)
