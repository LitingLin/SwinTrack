from data.operator.bbox.spatial.vectorized.torch.xyxy_to_cxcywh import box_xyxy_to_cxcywh
from data.types.bounding_box_format import BoundingBoxFormat
import torch
from typing import Optional
from data.tracking._common import _get_bounding_box_format, _get_bounding_box_normalization_helper


class FeatMapTargetIndicesGenerator:
    def __init__(self, search_feat_size, search_region_size):
        self.scale = torch.tensor(search_feat_size, dtype=torch.float64) / torch.tensor(search_region_size, dtype=torch.float64)
        feat_map_indices = torch.arange(0, search_feat_size[0] * search_feat_size[1], dtype=torch.long)
        self.feat_map_indices = feat_map_indices.reshape(search_feat_size[1], search_feat_size[0])

    def __call__(self, target_bbox: torch.Tensor):
        target_bbox_feat_indices = target_bbox.clone()
        target_bbox_feat_indices[::2] = target_bbox[::2] * self.scale[0]
        target_bbox_feat_indices[1::2] = target_bbox[1::2] * self.scale[1]
        target_bbox_feat_indices = target_bbox_feat_indices.to(torch.long)
        assert torch.all(target_bbox_feat_indices >= 0), f'target_bbox: {target_bbox}, target_bbox_feat_indices: {target_bbox_feat_indices}'
        target_bbox_feat_indices = self.feat_map_indices[target_bbox_feat_indices[1]: target_bbox_feat_indices[3] + 1, target_bbox_feat_indices[0]: target_bbox_feat_indices[2] + 1].flatten().clone()
        assert len(target_bbox_feat_indices) != 0
        return target_bbox_feat_indices


def get_target_feat_map_indices_single(_):
    return torch.tensor([0], dtype=torch.long)


def generate_target_class_vector(search_feat_size, target_feat_map_indices: Optional[torch.Tensor]):
    target_class_vector = torch.ones([search_feat_size[0] * search_feat_size[1]], dtype=torch.long)
    if target_feat_map_indices is not None:
        target_class_vector[target_feat_map_indices] = 0
    return target_class_vector


def generate_target_class_vector_one_as_positive(search_feat_size, target_feat_map_indices: Optional[torch.Tensor]):
    target_class_vector = torch.zeros([search_feat_size[0] * search_feat_size[1]], dtype=torch.float)
    if target_feat_map_indices is not None:
        target_class_vector[target_feat_map_indices] = 1
    return target_class_vector


def generate_target_bounding_box_label_matrix(bbox: torch.Tensor, search_region_size, target_feat_map_indices: torch.Tensor, bounding_box_format: BoundingBoxFormat, bbox_normalizer):
    length = len(target_feat_map_indices)

    if bounding_box_format == BoundingBoxFormat.CXCYWH:
        bbox = box_xyxy_to_cxcywh(bbox)
    bbox = bbox_normalizer.normalize(bbox, search_region_size)
    bbox = bbox.to(torch.float)
    return bbox.repeat(length, 1)


def label_generation(bbox, search_feat_size, search_region_size,
                     label_generation_fn, positive_sample_assignment_fn,
                     bounding_box_format, bbox_normalizer):
    target_feat_map_indices = positive_sample_assignment_fn(bbox)
    target_class_label_vector = label_generation_fn(search_feat_size, target_feat_map_indices)
    target_bounding_box_label_matrix = generate_target_bounding_box_label_matrix(bbox, search_region_size, target_feat_map_indices, bounding_box_format, bbox_normalizer)

    return target_feat_map_indices, target_class_label_vector, target_bounding_box_label_matrix


def negative_label_generation(search_feat_size, label_fn):
    return None, label_fn(search_feat_size, None), None


class TransTLabelGenerator:
    def __init__(self, search_feat_size, search_region_size,
                 label_one_as_positive: bool,
                 positive_label_assignment_method,
                 target_bounding_box_format,
                 bounding_box_normalization_helper):
        if label_one_as_positive:
            self.label_function = generate_target_class_vector_one_as_positive
        else:
            self.label_function = generate_target_class_vector
        self.search_feat_size = search_feat_size
        self.search_region_size = search_region_size
        self.target_bounding_box_format = target_bounding_box_format
        assert target_bounding_box_format in (BoundingBoxFormat.XYXY, BoundingBoxFormat.CXCYWH)
        assert positive_label_assignment_method in ('round',)
        if search_feat_size[0] * search_feat_size[1] == 1:
            self.positive_sample_assignment_fn = get_target_feat_map_indices_single
        elif positive_label_assignment_method == 'round':
            self.positive_sample_assignment_fn = FeatMapTargetIndicesGenerator(search_feat_size, search_region_size)
        else:
            raise NotImplementedError
        self.bounding_box_normalization_helper = bounding_box_normalization_helper

    def __call__(self, bbox, is_positive):
        if is_positive:
            return label_generation(bbox, self.search_feat_size, self.search_region_size,
                                    self.label_function, self.positive_sample_assignment_fn,
                                    self.target_bounding_box_format, self.bounding_box_normalization_helper)
        else:
            return negative_label_generation(self.search_feat_size, self.label_function)


def _batch_collate_target_feat_map_indices(target_feat_map_indices_list):
    batch_ids = []
    batch_target_feat_map_indices = []
    num_boxes_pos = 0
    for index, target_feat_map_indices in enumerate(target_feat_map_indices_list):
        if target_feat_map_indices is None:
            continue
        batch_ids.extend([index for _ in range(len(target_feat_map_indices))])
        batch_target_feat_map_indices.append(target_feat_map_indices)
        num_boxes_pos += len(target_feat_map_indices)

    num_boxes_pos = torch.as_tensor([num_boxes_pos], dtype=torch.float)
    if len(batch_ids) != 0:
        return torch.tensor(batch_ids, dtype=torch.long), torch.cat(batch_target_feat_map_indices), num_boxes_pos
    else:
        return None, None, num_boxes_pos


def TransT_label_collator(label_list):
    target_feat_map_indices_list = []
    target_class_label_vector_list = []
    target_bounding_box_label_matrix_list = []

    for target_feat_map_indices, target_class_label_vector, target_bounding_box_label_matrix in label_list:
        target_feat_map_indices_list.append(target_feat_map_indices)
        target_class_label_vector_list.append(target_class_label_vector)
        if target_bounding_box_label_matrix is not None:
            target_bounding_box_label_matrix_list.append(target_bounding_box_label_matrix)

    target_feat_map_indices_batch_id_vector, target_feat_map_indices_batch, num_boxes_pos = _batch_collate_target_feat_map_indices(target_feat_map_indices_list)
    target_class_label_vector_batch = torch.stack(target_class_label_vector_list)
    if len(target_bounding_box_label_matrix_list) != 0:
        target_bounding_box_label_matrix_batch = torch.cat(target_bounding_box_label_matrix_list, dim=0)
    else:
        target_bounding_box_label_matrix_batch = None
    collated = {'num_positive_samples': num_boxes_pos, 'class_label': target_class_label_vector_batch}
    if num_boxes_pos != 0:
        collated.update({
            'positive_sample_batch_dim_index': target_feat_map_indices_batch_id_vector,
            'positive_sample_feature_map_dim_index': target_feat_map_indices_batch,
            'bounding_box_label': target_bounding_box_label_matrix_batch
        })
    return collated


def build_label_generator_and_batch_collator(network_config, one_as_positive, multi_scale_with_wrapper=True):
    data_bounding_box_parameters = network_config['data']['bounding_box']
    assert data_bounding_box_parameters['format'] == 'CXCYWH'

    label_config = network_config['head']['output_protocol']['parameters']['label']

    positive_label_assignment_method = 'round'
    if 'label' in label_config:
        positive_label_assignment_method = label_config['positive_samples_assignment_method']
    search_size = network_config['data']['search_size']
    if 'scales' not in label_config:
        label_generator = TransTLabelGenerator(label_config['size'],
                                               search_size,
                                               one_as_positive, positive_label_assignment_method,
                                               _get_bounding_box_format(network_config),
                                               _get_bounding_box_normalization_helper(network_config))
        label_batch_collator = TransT_label_collator
    else:
        scales_parameters = label_config['scales']
        single_scale_generators = [TransTLabelGenerator(scale_parameters['size'],
                                                        search_size,
                                                        one_as_positive, positive_label_assignment_method,
                                                        _get_bounding_box_format(network_config),
                                                        _get_bounding_box_normalization_helper(network_config))
                                   for scale_parameters in scales_parameters]
        if multi_scale_with_wrapper:
            from data.tracking.label.multi_scale import SimpleMultiScaleLabelGeneratorWrapper, SimpleMultiScaleLabelBatchCollator
            label_generator = SimpleMultiScaleLabelGeneratorWrapper(single_scale_generators)
            label_batch_collator = SimpleMultiScaleLabelBatchCollator(TransT_label_collator)
        else:
            label_generator = single_scale_generators
            label_batch_collator = TransT_label_collator

    return label_generator, label_batch_collator
