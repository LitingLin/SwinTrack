import torch

from data.operator.bbox.spatial.vectorized.torch.cxcywh_to_xyxy import box_cxcywh_to_xyxy
from data.operator.bbox.spatial.vectorized.torch.intersection import bbox_compute_intersection_vectorized
from data.operator.bbox.spatial.vectorized.torch.utility.half_pixel_offset.image import bbox_restrict_in_image_boundary_
from data.operator.bbox.spatial.vectorized.torch.validity import bbox_is_valid_vectorized
from data.operator.bbox.spatial.vectorized.torch.xyxy_to_cxcywh import box_xyxy_to_cxcywh


def _get_scaling_and_translation_parameters(bounding_box, area_factor, output_size):
    curation_parameter = torch.empty([3, 2], dtype=torch.float64)
    scaling_factor, source_center, target_center = curation_parameter.unbind(0)

    w = bounding_box[2] - bounding_box[0]
    h = bounding_box[3] - bounding_box[1]

    background_size = ((w + h) * 0.5) * (area_factor - 1)

    w_z = w + background_size
    h_z = h + background_size
    scaling_factor[:] = ((output_size[0] * output_size[1]) / (w_z * h_z)).sqrt()

    source_center[0] = (bounding_box[0] + bounding_box[2]) / 2
    source_center[1] = (bounding_box[1] + bounding_box[3]) / 2

    target_center[0] = output_size[0] / 2
    target_center[1] = output_size[1] / 2

    return curation_parameter


def _adjust_bbox_size(bounding_box, min_wh):
    bounding_box = box_xyxy_to_cxcywh(bounding_box)
    torch.clamp_(bounding_box[2], min=min_wh[0])
    torch.clamp_(bounding_box[3], min=min_wh[1])
    return box_cxcywh_to_xyxy(bounding_box)


def _check_is_bounding_box_valid(bounding_box, image_size):
    if not (bounding_box[0] < bounding_box[2] and bounding_box[1] < bounding_box[3]):
        return False

    image_bounding_box = torch.zeros((4, ), dtype=torch.float64)
    image_bounding_box[2: 4] = image_size

    return bbox_is_valid_vectorized(bbox_compute_intersection_vectorized(bounding_box, image_bounding_box))


class SiamFCCurationParameterSimpleProvider:
    def __init__(self, area_factor, min_object_size=None):
        self.area_factor = area_factor
        self.cached_bounding_box = torch.empty((4, ), dtype=torch.float64)
        self.min_object_size = min_object_size

    def initialize(self, bounding_box):
        assert bounding_box[0] < bounding_box[2] and bounding_box[1] < bounding_box[3]
        self.cached_bounding_box[:] = bounding_box

    def get(self, curated_image_size):
        bounding_box = self.cached_bounding_box
        if self.min_object_size is not None:
            bounding_box = _adjust_bbox_size(bounding_box, self.min_object_size)
        curation_parameter = _get_scaling_and_translation_parameters(bounding_box, self.area_factor, curated_image_size)
        assert not torch.any(torch.isnan(curation_parameter))
        return curation_parameter

    def update(self, predicted_iou, predicted_bounding_box, image_size):
        assert image_size[0] > 0 and image_size[1] > 0
        predicted_bounding_box = predicted_bounding_box.clone()
        bbox_restrict_in_image_boundary_(predicted_bounding_box, image_size)
        if _check_is_bounding_box_valid(predicted_bounding_box, image_size):
            self.cached_bounding_box[:] = predicted_bounding_box
