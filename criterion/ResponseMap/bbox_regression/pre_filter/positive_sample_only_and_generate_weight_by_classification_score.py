import torch


def positive_only_generate_weight_by_classification_score_sample_filter(predicted, label, context):
    '''
        cls_score: (N, C, H, W)
        predicted_bbox: (N, H, W, 4)
        bbox_distribution: (N, H, W, 4 * (reg_max + 1))
    '''
    updated_predicted = {
        'bbox': predicted['bbox']
    }
    if 'bbox_gfocal' in predicted:
        updated_predicted['bbox_gfocal'] = predicted['bbox_gfocal']
    if 'bbox_coarse' in predicted:
        updated_predicted['bbox_coarse'] = predicted['bbox_coarse']
    if 'bounding_box_label' not in label:
        context['sample_weight'] = torch.zeros((1,), device=predicted['bbox'].device)
        return updated_predicted, None
    else:
        positive_sample_batch_dim_index = label['positive_sample_batch_dim_index']
        positive_sample_feature_map_dim_index = label['positive_sample_feature_map_dim_index']

        cls_score = predicted['class_score']
        weight_targets = cls_score.detach().flatten(2).transpose(1, 2).flatten(1)
        weight_targets = weight_targets[positive_sample_batch_dim_index, positive_sample_feature_map_dim_index].flatten()
        context['sample_weight'] = weight_targets

        predicted_bbox = updated_predicted['bbox']
        N, H, W, _ = predicted_bbox.shape
        predicted_bbox = predicted_bbox.view(N, H * W, 4)
        predicted_bbox = predicted_bbox[positive_sample_batch_dim_index, positive_sample_feature_map_dim_index]

        updated_predicted['bbox'] = predicted_bbox

        if 'bbox_coarse' in updated_predicted:
            predicted_bbox_coarse = updated_predicted['bbox_coarse']
            N, H, W, _ = predicted_bbox_coarse.shape
            predicted_bbox_coarse = predicted_bbox_coarse.view(N, H * W, 4)
            predicted_bbox_coarse = predicted_bbox_coarse[positive_sample_batch_dim_index, positive_sample_feature_map_dim_index]
            updated_predicted['bbox_coarse'] = predicted_bbox_coarse

        if 'bbox_gfocal' in updated_predicted:
            bbox_distribution = updated_predicted['bbox_gfocal']
            assert bbox_distribution.shape[0] == N and bbox_distribution.shape[1] == H and bbox_distribution.shape[2] == W
            bbox_distribution = bbox_distribution.view(N, H * W, -1)
            bbox_distribution = bbox_distribution[positive_sample_batch_dim_index, positive_sample_feature_map_dim_index]
            updated_predicted['bbox_gfocal'] = bbox_distribution
        return updated_predicted, (label['num_positive_samples'], label['bounding_box_label'])


def build_data_filter(*_):
    return positive_only_generate_weight_by_classification_score_sample_filter
