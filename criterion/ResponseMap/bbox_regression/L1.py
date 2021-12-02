import torch.nn as nn
from criterion.common.reduction.default import build_loss_reduction_function
from data.operator.bbox.spatial.vectorized.torch.cxcywh_to_xyxy import box_cxcywh_to_xyxy


def l1_loss_data_adaptor(pred, label, _):
    predicted_bbox = pred['bbox']
    if label is None:
        return False, predicted_bbox.sum() * 0
    (num_boxes_pos, target_bounding_box_label_matrix) = label
    return True, (box_cxcywh_to_xyxy(predicted_bbox), box_cxcywh_to_xyxy(target_bounding_box_label_matrix))


def reduce_by_weight(loss, pred, label, context):
    return ((loss * context['sample_weight'].unsqueeze(-1).expand(-1, 4)).reshape(-1)).sum() / 4


def build_L1(loss_parameters, *_):
    l1_loss = nn.L1Loss(reduction='none')
    if 'reduce' in loss_parameters and loss_parameters['reduce'] == 'weighted':
        loss_reduce_function = reduce_by_weight
    else:
        loss_reduce_function = build_loss_reduction_function(loss_parameters)
    return l1_loss, l1_loss_data_adaptor, loss_reduce_function
