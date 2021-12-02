import torch
from data.operator.bbox.spatial.xyxy2xywh import bbox_xyxy2xywh
import math
from data.operator.bbox.spatial.center import bbox_get_center_point
from data.operator.bbox.spatial.utility.half_pixel_offset.image import get_image_center_point, bounding_box_is_intersect_with_image, bounding_box_is_in_image_boundary
from data.operator.bbox.spatial.scale_and_translate import bbox_scale_and_translate
from data.operator.image_and_bbox.half_pixel_center.vectorized.pytorch.scale_and_translate import torch_scale_and_translate_half_pixel_offset
from data.operator.image.pytorch.mean import get_image_mean


def get_jittered_scaling_and_translate_factor(bbox, scaling, scaling_jitter_factor, translation_jitter_factor):
    scaling = scaling / torch.exp(torch.randn(2) * scaling_jitter_factor)
    bbox = bbox_xyxy2xywh(bbox)
    max_translate = (torch.tensor(bbox[2:4]) * scaling).sum() * 0.5 * translation_jitter_factor
    translate = (torch.rand(2) - 0.5) * max_translate
    return scaling, translate


def get_scaling_factor_from_area_factor(bbox, area_factor, output_size):
    bbox = bbox_xyxy2xywh(bbox)
    w, h = bbox[2: 4]
    w_z = w + (area_factor - 1) * ((w + h) * 0.5)
    h_z = h + (area_factor - 1) * ((w + h) * 0.5)
    scaling = math.sqrt((output_size[0] * output_size[1]) / (w_z * h_z))
    return torch.tensor((scaling, scaling), dtype=torch.float64)


def get_scaling_and_translation_parameters(bbox, area_factor, output_size):
    scaling = get_scaling_factor_from_area_factor(bbox, area_factor, output_size)

    source_center = bbox_get_center_point(bbox)
    target_center = get_image_center_point(output_size)
    source_center = torch.tensor(source_center)
    target_center = torch.tensor(target_center)
    return scaling, source_center, target_center


def prepare_SiamFC_curation_with_position_augmentation(bbox, area_factor, output_size, scaling_jitter_factor, translation_jitter_factor):
    while True:
        scaling = get_scaling_factor_from_area_factor(bbox, area_factor, output_size)
        scaling, translate = get_jittered_scaling_and_translate_factor(bbox, scaling, scaling_jitter_factor,
                                                                       translation_jitter_factor)

        source_center = bbox_get_center_point(bbox)
        target_center = get_image_center_point(output_size)
        target_center = (torch.tensor(target_center) - translate)

        output_bbox = bbox_scale_and_translate(bbox, scaling, source_center, target_center)

        if bounding_box_is_intersect_with_image(output_bbox, output_size):
            break
    source_center = torch.tensor(source_center)
    output_bbox = torch.tensor(output_bbox)
    curation_parameter = torch.stack((scaling, source_center, target_center))

    return curation_parameter, output_bbox


def prepare_SiamFC_curation(bbox, area_factor, output_size):
    curation_scaling, curation_source_center_point, curation_target_center_point = get_scaling_and_translation_parameters(bbox, area_factor, output_size)
    output_bbox = bbox_scale_and_translate(bbox, curation_scaling, curation_source_center_point, curation_target_center_point)
    output_bbox = torch.tensor(output_bbox)

    curation_parameter = torch.stack((curation_scaling, curation_source_center_point, curation_target_center_point))

    return curation_parameter, output_bbox


def do_SiamFC_curation(image, output_size, curation_parameter, interpolation_mode, image_mean=None, out_img=None, out_image_mean=None):
    if image_mean is None:
        image_mean = get_image_mean(image, out_image_mean)
    else:
        if out_image_mean is not None:
            out_image_mean[:] = image_mean
    output_image, _ = torch_scale_and_translate_half_pixel_offset(image, output_size, curation_parameter[0], curation_parameter[1], curation_parameter[2], image_mean, interpolation_mode, out_img)
    return output_image, image_mean
