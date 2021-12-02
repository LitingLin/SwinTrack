import torch
from data.tracking.methods.SiamFC.common.siamfc_curation import do_SiamFC_curation
import numpy as np
from torchvision.transforms import Grayscale
from torch.utils.data.dataloader import default_collate
from .pipeline import SiamTracker_training_prepare_SiamFC_curation, build_SiamTracker_image_augmentation_transformer


class SiamTrackerProcessor:
    def __init__(self,
                 template_size, search_size,
                 template_area_factor, search_area_factor,
                 template_scale_jitter_factor, search_scale_jitter_factor,
                 template_translation_jitter_factor, search_translation_jitter_factor,
                 gray_scale_probability,
                 color_jitter, label_generator, interpolation_mode):
        self.template_size = template_size
        self.search_size = search_size
        self.template_area_factor = template_area_factor
        self.search_area_factor = search_area_factor
        self.template_scale_jitter_factor = template_scale_jitter_factor
        self.search_scale_jitter_factor = search_scale_jitter_factor
        self.template_translation_jitter_factor = template_translation_jitter_factor
        self.search_translation_jitter_factor = search_translation_jitter_factor
        self.gray_scale_probability = gray_scale_probability
        self.interpolation_mode = interpolation_mode
        self.transform = build_SiamTracker_image_augmentation_transformer(color_jitter, True)
        self.gray_scale_transformer = Grayscale(3)
        self.label_generator = label_generator

    def __call__(self, z_image, z_bbox, x_image, x_bbox, is_positive):
        data = {}
        data['is_positive'] = is_positive
        if z_image is x_image:
            z_image = x_image = z_image.to(torch.float)
        else:
            z_image = z_image.to(torch.float)
            x_image = x_image.to(torch.float)

        z_curated_bbox, z_curation_parameter = SiamTracker_training_prepare_SiamFC_curation(
            z_bbox, self.template_area_factor,
            self.template_size,
            self.template_scale_jitter_factor,
            self.template_translation_jitter_factor)

        x_curated_bbox, x_curation_parameter = SiamTracker_training_prepare_SiamFC_curation(
            x_bbox, self.search_area_factor,
            self.search_size,
            self.search_scale_jitter_factor,
            self.search_translation_jitter_factor)

        z_curated_image, _ = do_SiamFC_curation(z_image, self.template_size, z_curation_parameter, self.interpolation_mode)
        x_curated_image, _ = do_SiamFC_curation(x_image, self.search_size, x_curation_parameter, self.interpolation_mode)

        z_curated_image /= 255.
        x_curated_image /= 255.

        if np.random.random() < self.gray_scale_probability:
            z_curated_image = self.gray_scale_transformer(z_curated_image)
            x_curated_image = self.gray_scale_transformer(x_curated_image)

        z_curated_image = self.transform(z_curated_image)
        x_curated_image = self.transform(x_curated_image)

        data.update({
            'z_curated_image': z_curated_image,
            'x_curated_image': x_curated_image,
        })

        labels = self.label_generator(x_curated_bbox, is_positive)

        data['label'] = labels

        return data


def _collate(batch_list, key_name, as_tensor):
    collated = tuple(element[key_name] for element in batch_list)
    if as_tensor:
        collated = default_collate(collated)
    return collated


def _collate_dict(batch_list, key_name, as_tensor, collated_dict):
    collated = _collate(batch_list, key_name, as_tensor)
    if collated_dict is None:
        return {key_name: collated}
    else:
        collated_dict[key_name] = collated
        return collated_dict


class SiamFCBatchDataCollator:
    def __init__(self, label_collator):
        self.label_collator = label_collator

    def __call__(self, data_list):
        samples_batched_on_device = None
        miscellanies_on_host = None
        miscellanies_on_device = None
        first_element = data_list[0]
        if 'z' in first_element:
            miscellanies_on_host = _collate_dict(data_list, 'z', False, miscellanies_on_host)
            miscellanies_on_host = _collate_dict(data_list, 'x', False, miscellanies_on_host)
            miscellanies_on_host = _collate_dict(data_list, 'z_bbox', True, miscellanies_on_host)
            miscellanies_on_host = _collate_dict(data_list, 'x_bbox', True, miscellanies_on_host)

        miscellanies_on_host = _collate_dict(data_list, 'is_positive', True, miscellanies_on_host)
        if 'z_augmented_image' in first_element:
            miscellanies_on_device = _collate_dict(data_list, 'z_augmented_image', False, miscellanies_on_device)
            miscellanies_on_device = _collate_dict(data_list, 'x_augmented_image', False, miscellanies_on_device)
            miscellanies_on_host = _collate_dict(data_list, 'z_curated_bbox', True, miscellanies_on_host)
            miscellanies_on_host = _collate_dict(data_list, 'z_curation_parameter', True, miscellanies_on_host)
            miscellanies_on_host = _collate_dict(data_list, 'x_curated_bbox', True, miscellanies_on_host)
            miscellanies_on_host = _collate_dict(data_list, 'x_curation_parameter', True, miscellanies_on_host)
        if 'z_curated_image' in first_element:
            z_curated_image = _collate(data_list, 'z_curated_image', True)
            x_curated_image = _collate(data_list, 'x_curated_image', True)
            samples_batched_on_device = z_curated_image, x_curated_image

        labels_batched_on_device = self.label_collator(tuple(element['label'] for element in data_list))
        return samples_batched_on_device, labels_batched_on_device, miscellanies_on_host, miscellanies_on_device
