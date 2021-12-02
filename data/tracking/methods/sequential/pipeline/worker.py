import torch
import torchvision
from data.tracking.methods.SiamFC.common.siamfc_curation import prepare_SiamFC_curation, do_SiamFC_curation
from .common import get_transform
import time


def _decode_image(image_path):
    image = torchvision.io.image.read_image(image_path, torchvision.io.image.ImageReadMode.RGB)
    image = image.to(torch.float)
    image /= 255.
    return image


class SequentialWorkerDataPipeline:
    def __init__(self, batch_size, template_area_factor, curated_template_image_size, interpolate_mode):
        self.template_area_factor = template_area_factor
        self.curated_template_image_size = curated_template_image_size
        self.interpolate_mode = interpolate_mode
        self.sequence_uuid = [None] * batch_size
        self.transform = get_transform()

    @staticmethod
    def _get_dummy_bbox(image_size):
        return torch.tensor([image_size[0] * 0.25, image_size[1] * 0.25, image_size[0] * 0.75, image_size[1] * 0.75], dtype=torch.float64)

    def __call__(self, index_element, sequence_uuid, sequence_name, sequence_dataset_name, frame_index, sequence_length, template_image_path, template_target_bbox, search_image_path, search_target_bbox):
        curated_first_frame_image = None
        template_image_mean = None
        first_frame_begin_time = None
        if self.sequence_uuid[index_element] != sequence_uuid:
            first_frame_begin_time = time.perf_counter()
            template_image = _decode_image(template_image_path)
            curation_parameter, _ = prepare_SiamFC_curation(template_target_bbox, self.template_area_factor, self.curated_template_image_size)
            curated_first_frame_image, template_image_mean = do_SiamFC_curation(template_image, self.curated_template_image_size, curation_parameter, self.interpolate_mode)
            curated_first_frame_image = self.transform(curated_first_frame_image)
            self.sequence_uuid[index_element] = sequence_uuid

            search_begin_time = time.perf_counter()
            template_target_bbox = torch.tensor(template_target_bbox)
            if search_image_path == template_image_path:
                search_image = template_image
            else:
                search_image = _decode_image(search_image_path)
        else:
            search_begin_time = time.perf_counter()
            search_image = _decode_image(search_image_path)
            template_target_bbox = None

        search_image_size = search_image.shape[1:]
        search_image_size = torch.tensor([search_image_size[1], search_image_size[0]])
        search_object_existence = search_target_bbox is not None

        if search_target_bbox is not None:
            search_target_bbox = torch.tensor(search_target_bbox)
        if not search_object_existence:
            search_target_bbox = self._get_dummy_bbox(search_image_size)

        return sequence_uuid, sequence_name, sequence_dataset_name, frame_index, sequence_length, \
               curated_first_frame_image, template_image_mean, template_target_bbox, \
               search_image, search_object_existence, search_target_bbox, \
               first_frame_begin_time, search_begin_time


class SequentialWorkerDataProcessor:
    def __init__(self, batch_size, template_area_factor, curated_template_image_size, interpolate_mode):
        self.data_processing_pipeline = SequentialWorkerDataPipeline(batch_size, template_area_factor, curated_template_image_size, interpolate_mode)

    def __call__(self, index, data):
        if data is None:
            return None
        sequence_uuid, sequence_name, sequence_dataset_name, frame_index, sequence_length, (template_image_path, template_target_bbox), (current_search_image_path, current_search_image_target_bbox) = data
        return self.data_processing_pipeline(index, sequence_uuid, sequence_name, sequence_dataset_name, frame_index, sequence_length, template_image_path, template_target_bbox, current_search_image_path, current_search_image_target_bbox)


def _collate_as_torch_tensor(batch_list, index):
    tensors = tuple(elem[index] for elem in batch_list)
    if isinstance(tensors[0], torch.Tensor):
        return torch.stack(tensors, dim=0)
    else:
        return torch.tensor(tensors)


def _collate_as_tuple(batch_list, index):
    return tuple(elem[index] for elem in batch_list)


def _collate_auto_type(batch_list, index):
    tensors = tuple(elem[index] for elem in batch_list)
    has_none = False
    for tensor in tensors:
        if tensor is None:
            has_none = True
    if not has_none:
        if isinstance(tensors[0], torch.Tensor):
            return torch.stack(tensors, dim=0)
        else:
            return torch.tensor(tensors)
    return tensors


class SequentialSamplingWorkerDataCollator:
    def __init__(self, time_keeping_host_only=True):
        self.time_keeping_host_only = time_keeping_host_only

    def __call__(self, batch_list):
        batch_list = [elem for elem in batch_list if elem is not None]
        if len(batch_list) == 0:
            return None, None, None, None
        sample = {}
        label = {}
        misc_on_host = {}
        misc_on_device = {}

        sequence_uuid_batch = _collate_as_tuple(batch_list, 0)
        sequence_name_batch = _collate_as_tuple(batch_list, 1)
        sequence_dataset_name_batch = _collate_as_tuple(batch_list, 2)
        frame_index_batch = _collate_as_torch_tensor(batch_list, 3)
        sequence_length_batch = _collate_as_torch_tensor(batch_list, 4)
        curated_template_image_batch = _collate_as_tuple(batch_list, 5)
        template_image_mean_batch = _collate_auto_type(batch_list, 6)
        template_target_bbox_batch = _collate_auto_type(batch_list, 7)
        search_image_batch = _collate_as_tuple(batch_list, 8)
        search_object_existence_batch = _collate_as_torch_tensor(batch_list, 9)
        search_target_bbox_batch = _collate_auto_type(batch_list, 10)

        misc_on_host['seq_uuid'] = sequence_uuid_batch
        misc_on_host['seq_name'] = sequence_name_batch
        misc_on_host['seq_dataset_name'] = sequence_dataset_name_batch
        misc_on_host['seq_len'] = sequence_length_batch
        sample['z_curated'] = curated_template_image_batch
        sample['x'] = search_image_batch
        misc_on_host['z_bbox'] = template_target_bbox_batch
        misc_on_device['z_image_mean'] = template_image_mean_batch
        misc_on_host['frame_index'] = frame_index_batch
        label['target_existence'] = search_object_existence_batch
        label['target_bbox'] = search_target_bbox_batch

        if not self.time_keeping_host_only:
            template_branch_begin_time_batch = _collate_as_tuple(batch_list, 11)
            search_branch_begin_time_batch = _collate_as_torch_tensor(batch_list, 12)

            misc_on_host['1st_begin_t'] = template_branch_begin_time_batch
            misc_on_host['x_begin_t'] = search_branch_begin_time_batch
        return sample, label, misc_on_host, misc_on_device
