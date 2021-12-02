from data.operator.bbox.spatial.vectorized.torch.utility.normalize import BoundingBoxNormalizationHelper
from data.types.bounding_box_format import BoundingBoxFormat


def _get_bounding_box_normalization_helper(network_config: dict):
    bounding_box_config = network_config['data']['bounding_box']
    return BoundingBoxNormalizationHelper(
        bounding_box_config['normalization_protocol']['interval'],
        bounding_box_config['normalization_protocol']['range'],
    )


def _get_bounding_box_format(network_config: dict):
    bounding_box_config = network_config['data']['bounding_box']
    return BoundingBoxFormat[bounding_box_config['format']]
