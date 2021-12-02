from data.operator.bbox.spatial.vectorized.torch.scale_and_translate import bbox_scale_and_translate_vectorized
from data.types.bounding_box_format import BoundingBoxFormat


class DefaultBoundingBoxPostProcessor:
    def __init__(self, search_region_size, bbox_normalizer, input_format=BoundingBoxFormat.XYXY):
        self.search_region_size = search_region_size
        assert input_format in (BoundingBoxFormat.XYXY, BoundingBoxFormat.CXCYWH)
        self.input_format = input_format
        self.bbox_normalizer = bbox_normalizer

    def __call__(self, bbox_normalized, curation_parameter):
        assert curation_parameter.ndim in (2, 3)
        curation_scaling, curation_source_center_point, curation_target_center_point = curation_parameter.unbind(dim=-2)
        bbox = self.bbox_normalizer.denormalize(bbox_normalized, self.search_region_size)
        if self.input_format == BoundingBoxFormat.CXCYWH:
            from data.operator.bbox.spatial.vectorized.torch.cxcywh_to_xyxy import box_cxcywh_to_xyxy
            bbox = box_cxcywh_to_xyxy(bbox)

        bbox = bbox_scale_and_translate_vectorized(bbox, 1.0 / curation_scaling, curation_target_center_point, curation_source_center_point)
        return bbox


from data.tracking._common import _get_bounding_box_normalization_helper, _get_bounding_box_format


def build_bounding_box_post_processor(network_config: dict):
    return DefaultBoundingBoxPostProcessor(network_config['data']['search_size'], _get_bounding_box_normalization_helper(network_config), _get_bounding_box_format(network_config))
