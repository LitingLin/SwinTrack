import torch
from data.operator.bbox.spatial.vectorized.torch.cxcywh_to_xyxy import box_cxcywh_to_xyxy
from criterion.common.reduction.default import build_loss_reduction_function


class VarifocalDataAdaptor:
    def __init__(self, quality_fn, quality_fn_param_predicted, quality_fn_param_label):
        self.quality_fn = quality_fn
        self.quality_fn_param_predicted = quality_fn_param_predicted
        self.quality_fn_param_label = quality_fn_param_label

    def __call__(self, predicted, label, _):
        cls_score, predicted_bbox = predicted['class_score'], predicted[self.quality_fn_param_predicted]

        N, num_classes, H, W = cls_score.shape
        assert num_classes == 1

        cls_score = cls_score.flatten(0)
        if self.quality_fn is not None:
            quality_score = torch.zeros((N, H * W), dtype=cls_score.dtype, device=cls_score.device)

            if 'bounding_box_label' in label:
                positive_sample_batch_dim_index = label['positive_sample_batch_dim_index']
                positive_sample_feature_map_dim_index = label['positive_sample_feature_map_dim_index']
                predicted_bbox = predicted_bbox.view(N, H * W, 4)
                predicted_bbox = predicted_bbox[positive_sample_batch_dim_index, positive_sample_feature_map_dim_index]
                quality_score[
                    positive_sample_batch_dim_index, positive_sample_feature_map_dim_index] = \
                    self.quality_fn(box_cxcywh_to_xyxy(predicted_bbox.detach()),
                                    box_cxcywh_to_xyxy(label[self.quality_fn_param_label]))
            quality_score = quality_score.flatten(0)
            return True, (cls_score, quality_score)
        else:
            return True, (cls_score, label['class_label'].flatten(0))


def build_varifocal(loss_parameters: dict, *_):
    from criterion.modules.varifocal_loss import VarifocalLoss

    quality_function = loss_parameters['quality_target']['function']
    assert quality_function in ('IoU', 'none')
    if quality_function == 'IoU':
        from core.operator.iou.iou import iou
        quality_fn = iou
        quality_fn_param_predicted = loss_parameters['quality_target']['parameters']['predicted']
        quality_fn_param_label = loss_parameters['quality_target']['parameters']['label']
    else:
        quality_fn = None
        quality_fn_param_predicted = None
        quality_fn_param_label = None

    return VarifocalLoss(False, loss_parameters['alpha'], loss_parameters['gamma'], loss_parameters['iou_weighted']),\
           VarifocalDataAdaptor(quality_fn, quality_fn_param_predicted, quality_fn_param_label), build_loss_reduction_function(loss_parameters)
