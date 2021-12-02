import torch
from core.run.metric_logger.context import get_logger


class SiamFCMetricCollector:
    @staticmethod
    def pre_processing(samples, targets, miscellanies_on_host, miscellanies_on_device):
        positive_samples = miscellanies_on_host['is_positive']
        get_logger().log({'pos_samples_ratio': torch.sum(positive_samples) / len(positive_samples)})
        return samples, targets, miscellanies_on_host, miscellanies_on_device
