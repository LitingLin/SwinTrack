import torch
from data.tracking.methods.SiamFC.common.siamfc_curation import prepare_SiamFC_curation_with_position_augmentation, \
    do_SiamFC_curation
from data.operator.bbox.spatial.vectorized.torch.utility.half_pixel_offset.image import bbox_restrict_in_image_boundary_
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from torchvision import transforms


def build_SiamTracker_image_augmentation_transformer(color_jitter=0.4, imagenet_normalization=True):
    # color jitter is enabled when not using AA
    if isinstance(color_jitter, (list, tuple)):
        # color jitter should be a 3-tuple/list if spec brightness/contrast/saturation
        # or 4 if also augmenting hue
        assert len(color_jitter) in (3, 4)
    else:
        # if it's a scalar, duplicate for brightness, contrast, and saturation, no hue
        color_jitter = (float(color_jitter),) * 3
    transform_list = [
        transforms.ColorJitter(*color_jitter)
    ]
    if imagenet_normalization:
        transform_list += [
            transforms.Normalize(
                mean=torch.tensor(IMAGENET_DEFAULT_MEAN),
                std=torch.tensor(IMAGENET_DEFAULT_STD))
        ]
    return transforms.Compose(transform_list)


def SiamTracker_training_prepare_SiamFC_curation(bbox, area_factor, output_size, scaling_jitter_factor,
                                                 translation_jitter_factor):
    curation_parameter, bbox = \
        prepare_SiamFC_curation_with_position_augmentation(bbox, area_factor, output_size,
                                                           scaling_jitter_factor, translation_jitter_factor)
    bbox_restrict_in_image_boundary_(bbox, output_size)

    return bbox, curation_parameter
