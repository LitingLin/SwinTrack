import time
import torch
from typing import Dict, List
import uuid
import copy
from core.run.metric_logger.context import get_logger
from data.tracking.methods.SiamFC.common.siamfc_curation import do_SiamFC_curation
from collections import namedtuple
from .host_cache import UUIDBasedCacheService
from .common import get_transform

HostDataProcessorSequenceContext = namedtuple('HostProcessorSequenceContext', ('name', 'dataset_name', 'length', 'curation_parameter_provider', 'first_frame_target_bbox', 'begin_time_recording_array', 'end_time_recording_array'))
HostDataProcessorFrameContext = namedtuple('HostDataProcessorFrameContext', ('sequence_uuid', 'frame_index', 'image_size', 'target_existence', 'target_bbox'))


class EvaluationHostPipeline:
    def __init__(self, batch_size,
                 template_curation_image_size,
                 search_curation_image_size,  # W, H
                 search_curation_parameter_provider,
                 template_curated_image_feature_cache_service: UUIDBasedCacheService,
                 template_image_mean_cache_service: UUIDBasedCacheService,
                 hook, post_processor, bounding_box_post_processor,
                 interpolation_mode: str = 'bicubic',
                 time_keeping_host_only = True):
        self.search_curation_image_size = search_curation_image_size
        self.interpolation_mode = interpolation_mode
        self.search_curation_parameter_provider = search_curation_parameter_provider
        self.template_curated_image_feature_cache_service = template_curated_image_feature_cache_service
        self.template_image_mean_cache_service = template_image_mean_cache_service
        self.hook = hook
        self.post_processor = post_processor
        self.bounding_box_post_processor = bounding_box_post_processor
        self.curation_parameter_cache = torch.empty((batch_size, 3, 2), dtype=torch.float64)
        self.transform = get_transform()
        self.time_keeping_host_only = time_keeping_host_only

        self.template_curated_image_cache_shape = (batch_size, 3, template_curation_image_size[1], template_curation_image_size[0])
        self.search_curated_image_cache_shape = (batch_size, 3, self.search_curation_image_size[1], self.search_curation_image_size[0])
        self.device = torch.device('cpu')

        self.template_curated_image_cache = None
        self.search_curated_image_cache = None

    def on_device_changed(self, device):
        self.device = device
        if self.template_curated_image_cache is not None:
            self.template_curated_image_cache = self.template_curated_image_cache.to(device)
            self.search_curated_image_cache = self.search_curated_image_cache.to(device)
        self.template_curated_image_feature_cache_service.to(device)
        self.template_image_mean_cache_service.to(device)
        if hasattr(self.post_processor, 'to'):
            self.post_processor.to(device)

    def on_epoch_begin(self, epoch):
        assert self.template_curated_image_cache is None
        self.template_curated_image_cache = torch.empty(self.template_curated_image_cache_shape, dtype=torch.float, device=self.device)
        self.search_curated_image_cache = torch.empty(self.search_curated_image_cache_shape, dtype=torch.float, device=self.device)
        self.template_curated_image_feature_cache_service.create()
        self.template_image_mean_cache_service.create()
        self.tracking_list: Dict[uuid.UUID, HostDataProcessorSequenceContext] = {}
        self.processing_frame_contexts: List[HostDataProcessorFrameContext] = []
        self.initializing_uuids = []

    def on_epoch_end(self, epoch):
        self.template_curated_image_cache = None
        self.search_curated_image_cache = None
        self.template_curated_image_feature_cache_service.close()
        self.template_image_mean_cache_service.close()
        self.tracking_list = None
        self.processing_frame_contexts = None
        self.initializing_uuids = None

    def pre_initialization(self, samples, targets, miscellanies_on_host, miscellanies_on_device):
        if samples is None:
            get_logger().log(local={'batch_size': 0})
            return None

        assert len(self.processing_frame_contexts) == 0
        uuid_list = miscellanies_on_host['seq_uuid']  # prevent from collision
        assert len(uuid_list) > 0
        get_logger().log(local={'batch_size': len(uuid_list)})
        sequence_dataset_name_list = miscellanies_on_host['seq_dataset_name']
        sequence_name_list = miscellanies_on_host['seq_name']
        sequence_length_list = miscellanies_on_host['seq_len']
        curated_template_image_list = samples['z_curated']
        search_image_list = samples['x']
        template_object_bbox_list = miscellanies_on_host['z_bbox']

        image_mean_list = miscellanies_on_device['z_image_mean']

        frame_indices_list = miscellanies_on_host['frame_index']

        target_existing_flags = targets['target_existence']
        target_bounding_boxes = targets['target_bbox']

        if not self.time_keeping_host_only:
            first_frame_begin_time_batch = miscellanies_on_host['1st_begin_t']
            search_branch_begin_time_batch = miscellanies_on_host['x_begin_t']

        sequence_contexts = []

        for index_of_object, \
            (uuid, sequence_name, sequence_dataset_name, frame_index, sequence_length,
            template_curated_image, template_object_bbox, target_existence, target_bounding_box,
            template_image_mean, search_image) in enumerate(zip(
            uuid_list, sequence_name_list, sequence_dataset_name_list, frame_indices_list, sequence_length_list,
            curated_template_image_list, template_object_bbox_list, target_existing_flags, target_bounding_boxes,
            image_mean_list, search_image_list)):
            assert sequence_length > 1
            assert frame_index > 0
            if frame_index == 1:
                assert uuid not in self.tracking_list

                begin_time_recording_array = torch.empty((sequence_length,), dtype=torch.float64)
                end_time_recording_array = torch.empty((sequence_length, ), dtype=torch.float64)

                if self.time_keeping_host_only:
                    first_frame_begin_time = time.perf_counter()
                else:
                    first_frame_begin_time = first_frame_begin_time_batch[index_of_object]
                begin_time_recording_array[0] = first_frame_begin_time

                search_image_curation_parameter_provider = copy.deepcopy(self.search_curation_parameter_provider)
                search_image_curation_parameter_provider.initialize(template_object_bbox)
                assert template_curated_image is not None and template_image_mean is not None
                self.template_curated_image_cache[len(self.initializing_uuids), ...] = template_curated_image
                self.initializing_uuids.append(uuid)
                self.template_image_mean_cache_service.put(uuid, template_image_mean)

                self.tracking_list[uuid] = HostDataProcessorSequenceContext(sequence_name, sequence_dataset_name, sequence_length, search_image_curation_parameter_provider, template_object_bbox, begin_time_recording_array, end_time_recording_array)
            else:
                assert template_curated_image is None and template_image_mean is None
                template_image_mean = self.template_image_mean_cache_service.get(uuid)

            sequence_context = self.tracking_list[uuid]
            assert sequence_context.name == sequence_name and sequence_context.length == sequence_length

            if self.time_keeping_host_only:
                begin_time = time.perf_counter()
            else:
                begin_time = search_branch_begin_time_batch[index_of_object]
            sequence_context.begin_time_recording_array[frame_index] = begin_time

            search_image_size = search_image.shape[1:]
            search_image_size = torch.tensor((search_image_size[1], search_image_size[0]))  # (W, H)

            sequence_contexts.append(sequence_context)
            self.processing_frame_contexts.append(
                HostDataProcessorFrameContext(uuid, frame_index, search_image_size, target_existence,
                                              target_bounding_box))
            search_image_curation_parameter_provider = sequence_context.curation_parameter_provider

            curation_parameter = search_image_curation_parameter_provider.get(self.search_curation_image_size)
            self.curation_parameter_cache[index_of_object, :, :] = curation_parameter
            do_SiamFC_curation(search_image, self.search_curation_image_size, curation_parameter,
                               self.interpolation_mode, template_image_mean,
                               out_img=self.search_curated_image_cache[index_of_object, ...])

        if self.hook is not None:
            self.hook.pre_processing(sequence_contexts, self.processing_frame_contexts)

        self.cached_x = self.transform(self.search_curated_image_cache[:len(uuid_list), ...])
        if len(self.initializing_uuids) == 0:
            return None
        else:
            return self.template_curated_image_cache[:len(self.initializing_uuids), ...]

    def on_initialized(self, z_feats):
        if len(self.processing_frame_contexts) == 0:
            return None

        if z_feats is not None:
            self.template_curated_image_feature_cache_service.put_batch(self.initializing_uuids, z_feats)
            for initialized_uuid in self.initializing_uuids:
                self.tracking_list[initialized_uuid].end_time_recording_array[0] = time.perf_counter()

        x = self.cached_x
        self.initializing_uuids.clear()
        del self.cached_x
        sequence_uuids = tuple(context.sequence_uuid for context in self.processing_frame_contexts)

        return {'z_feat': self.template_curated_image_feature_cache_service.get_batch(sequence_uuids), 'x': x}

    def _post_processing_do_post_hook(self, predicted_bounding_boxes):
        hook_return = None
        if self.hook is not None:
            hook_return = self.hook.post_processing(predicted_bounding_boxes)
        self.processing_frame_contexts.clear()
        return hook_return

    def post_tracking(self, outputs):
        if len(self.processing_frame_contexts) == 0:
            return self._post_processing_do_post_hook(None)
        assert outputs is not None

        if self.post_processor is not None:
            outputs = self.post_processor(outputs)

        predicted_ious, predicted_bounding_boxes = outputs['conf'], outputs['bbox']
        assert len(self.processing_frame_contexts) == len(predicted_ious) == len(predicted_bounding_boxes)

        predicted_ious = predicted_ious.cpu()
        predicted_bounding_boxes = predicted_bounding_boxes.cpu()
        predicted_bounding_boxes = predicted_bounding_boxes.to(torch.float64)

        if self.bounding_box_post_processor is not None:
            predicted_bounding_boxes = self.bounding_box_post_processor(predicted_bounding_boxes, self.curation_parameter_cache[:predicted_bounding_boxes.shape[0], ...])

        for index in range(len(self.processing_frame_contexts)):
            sequence_frame_context = self.processing_frame_contexts[index]
            if sequence_frame_context is None:
                continue
            predicted_iou = predicted_ious[index]
            predicted_bounding_box = predicted_bounding_boxes[index]

            sequence_context = self.tracking_list[sequence_frame_context.sequence_uuid]
            sequence_context.end_time_recording_array[sequence_frame_context.frame_index] = time.perf_counter()
            is_last_frame = sequence_frame_context.frame_index + 1 == sequence_context.length

            if is_last_frame:
                uuid = sequence_frame_context.sequence_uuid
                if self.hook is not None:
                    self.hook.fill_time_cost_array(uuid, sequence_context.begin_time_recording_array, sequence_context.end_time_recording_array)
                del self.tracking_list[uuid]
                self.template_curated_image_feature_cache_service.release(uuid)
                self.template_image_mean_cache_service.release(uuid)
                continue

            curation_parameter_provider = sequence_context.curation_parameter_provider
            curation_parameter_provider.update(predicted_iou, predicted_bounding_box, sequence_frame_context.image_size)
        return self._post_processing_do_post_hook(predicted_bounding_boxes)
