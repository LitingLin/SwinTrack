import numpy as np
from .center_location_error import calculate_center_location_error_torch_vectorized
from data.operator.bbox.spatial.vectorized.iou import bbox_compute_iou_numpy_vectorized
from data.operator.bbox.spatial.vectorized.validity import bbox_is_valid_vectorized


class OPEEvaluationParameter:
    bins_of_center_location_error = 51
    bins_of_normalized_center_location_error = 51
    bins_of_intersection_of_union = 21


def _calc_curves(ious, center_errors, norm_center_errors, parameter: OPEEvaluationParameter = OPEEvaluationParameter):
    ious = np.asarray(ious, np.float64)[:, np.newaxis]
    center_errors = np.asarray(center_errors, np.float64)[:, np.newaxis]
    norm_center_errors = np.asarray(norm_center_errors, np.float64)[:, np.newaxis]

    thr_iou = np.linspace(0, 1, parameter.bins_of_intersection_of_union)[np.newaxis, :]
    thr_ce = np.arange(0, parameter.bins_of_center_location_error)[np.newaxis, :]
    thr_nce = np.linspace(0, 0.5, parameter.bins_of_normalized_center_location_error)[np.newaxis, :]

    bin_iou = np.greater_equal(ious, thr_iou)
    bin_ce = np.less_equal(center_errors, thr_ce)
    bin_nce = np.less_equal(norm_center_errors, thr_nce)

    succ_curve = np.mean(bin_iou, axis=0)
    prec_curve = np.mean(bin_ce, axis=0)
    norm_prec_curve = np.mean(bin_nce, axis=0)

    return succ_curve, prec_curve, norm_prec_curve


def _calc_ao_sr(ious):
    return np.mean(ious), np.mean(ious >= 0.5), np.mean(ious >= 0.75)


def calculate_evaluation_metrics(predicted_bounding_boxes, groundtruth_bounding_boxes, bounding_box_validity_flags,
                                 parameter: OPEEvaluationParameter = OPEEvaluationParameter):
    ious = bbox_compute_iou_numpy_vectorized(predicted_bounding_boxes, groundtruth_bounding_boxes)
    assert not (ious > 1.).any()

    groundtruth_bboxes_validity = bbox_is_valid_vectorized(groundtruth_bounding_boxes)
    if bounding_box_validity_flags is None:
        assert np.all(groundtruth_bboxes_validity)
    else:
        assert bounding_box_validity_flags.dtype == np.bool_
        # assert not np.any(np.bitwise_and(groundtruth_bboxes_validity, np.invert(bounding_box_validity_flags)))

    center_location_errors = calculate_center_location_error_torch_vectorized(predicted_bounding_boxes, groundtruth_bounding_boxes, False)
    normalized_center_location_errors = calculate_center_location_error_torch_vectorized(predicted_bounding_boxes, groundtruth_bounding_boxes, True)

    predicted_bboxes_validity = bbox_is_valid_vectorized(predicted_bounding_boxes)
    ious[~predicted_bboxes_validity] = -1.0
    center_location_errors[~predicted_bboxes_validity] = float('inf')
    normalized_center_location_errors[~predicted_bboxes_validity] = float('inf')

    if bounding_box_validity_flags is not None:
        ious = ious[bounding_box_validity_flags]
        center_location_errors = center_location_errors[bounding_box_validity_flags]
        normalized_center_location_errors = normalized_center_location_errors[bounding_box_validity_flags]

    ious[ious == 0] = -1.0

    succ_curve, prec_curve, norm_prec_curve = _calc_curves(ious, center_location_errors,
                                                           normalized_center_location_errors, parameter)
    ious[ious < 0.] = 0
    ao, sr_at_0_5, sr_at_0_75 = _calc_ao_sr(ious)

    return ao, sr_at_0_5, sr_at_0_75, succ_curve, prec_curve, norm_prec_curve
