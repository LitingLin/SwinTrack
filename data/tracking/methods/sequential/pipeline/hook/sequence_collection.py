import torch
import uuid
from typing import Dict, List
from ..evaluation_host import HostDataProcessorSequenceContext, HostDataProcessorFrameContext


class SequencePerformanceEvaluationContext:
    def __init__(self, object_existence, groundtruth_bboxes, predicted_bboxes, time_cost_array) -> None:
        self.object_existence = object_existence
        self.groundtruth_bboxes = groundtruth_bboxes
        self.predicted_bboxes = predicted_bboxes
        self.time_cost_array = time_cost_array


class SequenceTrackingPerformanceMetricCollector:
    def __init__(self):
        self.tracking_sequences: Dict[uuid.UUID, SequencePerformanceEvaluationContext] = {}
        self.processing_sequences: List[HostDataProcessorSequenceContext] = None
        self.processing_frames: List[HostDataProcessorFrameContext] = None
        self.processing_sequence_evaluation_contexts: List[SequencePerformanceEvaluationContext] = []

    def pre_processing(self, sequence_contexts: List[HostDataProcessorSequenceContext], current_frame_contexts: List[HostDataProcessorFrameContext]):
        assert self.processing_sequences is None and self.processing_frames is None and len(self.processing_sequence_evaluation_contexts) == 0
        for sequence_context, frame_context in zip(sequence_contexts, current_frame_contexts):
            assert sequence_context is not None
            if frame_context.frame_index == 1:
                object_existence = None
                if frame_context.target_existence is not None:
                    object_existence = torch.empty((sequence_context.length, ), dtype=torch.bool)
                    object_existence[0] = True
                groundtruth_bboxes = None
                if frame_context.target_bbox is not None:
                    groundtruth_bboxes = torch.empty((sequence_context.length, 4), dtype=torch.float64)
                    groundtruth_bboxes[0, :] = sequence_context.first_frame_target_bbox
                predicted_bboxes = torch.empty((sequence_context.length, 4), dtype=torch.float64)
                predicted_bboxes[0, :] = sequence_context.first_frame_target_bbox
                self.tracking_sequences[frame_context.sequence_uuid] = SequencePerformanceEvaluationContext(object_existence, groundtruth_bboxes, predicted_bboxes, None)
            evaluation_context = self.tracking_sequences[frame_context.sequence_uuid]
            if evaluation_context.object_existence is not None:
                evaluation_context.object_existence[frame_context.frame_index] = frame_context.target_existence
            if evaluation_context.groundtruth_bboxes is not None:
                evaluation_context.groundtruth_bboxes[frame_context.frame_index, :] = frame_context.target_bbox
            self.processing_sequence_evaluation_contexts.append(evaluation_context)
        self.processing_sequences = sequence_contexts
        self.processing_frames = current_frame_contexts

    def fill_time_cost_array(self, sequence_uuid, begin_time_array, end_time_array):
        self.tracking_sequences[sequence_uuid].time_cost_array = end_time_array - begin_time_array

    def _post_processing_post_cleanup(self):
        self.processing_sequences = None
        self.processing_frames = None
        self.processing_sequence_evaluation_contexts.clear()

    def post_processing(self, predicted_bounding_boxes):
        if predicted_bounding_boxes is None:
            self._post_processing_post_cleanup()
            return None
        full_sequences = []
        for sequence_context, frame_context, sequence_evaluation_context, predicted_bounding_box in zip(self.processing_sequences, self.processing_frames, self.processing_sequence_evaluation_contexts, predicted_bounding_boxes):
            if sequence_context is None:
                assert frame_context is None and sequence_evaluation_context is None
                continue
            sequence_evaluation_context.predicted_bboxes[frame_context.frame_index, :] = predicted_bounding_box
            if frame_context.frame_index + 1 == sequence_context.length:
                full_sequences.append((sequence_context.dataset_name, sequence_context.name, sequence_evaluation_context.object_existence, sequence_evaluation_context.groundtruth_bboxes, sequence_evaluation_context.predicted_bboxes, sequence_evaluation_context.time_cost_array))
                del self.tracking_sequences[frame_context.sequence_uuid]

        self._post_processing_post_cleanup()
        return full_sequences
