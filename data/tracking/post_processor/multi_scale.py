import numpy as np
import torch
from torch.utils.data.dataloader import default_collate


class MultiScalePostProcessor:
    def __init__(self, single_scale_processors):
        self.single_scale_processors = single_scale_processors

    def __call__(self, network_outputs):
        if network_outputs is None:
            return None
        outputs = [single_scale_processor(network_output) for single_scale_processor, network_output in zip(self.single_scale_processors, network_outputs)]
        outputs = default_collate(outputs)  # (S, N, 4), (S, N)
        confidence_scores = outputs['conf']
        bounding_boxes = outputs['bbox']
        N = confidence_scores.shape[1]
        confidence_scores, indices = torch.max(confidence_scores, 0)
        return {'bbox': bounding_boxes.transpose(0, 1)[torch.arange(N), indices, :], 'conf': confidence_scores}

    def to(self, device):
        for single_scale_processor in self.single_scale_processors:
            if hasattr(single_scale_processor, 'to'):
                single_scale_processor.to(device)
