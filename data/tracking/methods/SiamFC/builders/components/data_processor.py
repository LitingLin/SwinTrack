from data.tracking.label.builder import build_label_generator_and_batch_collator
from ...pipeline.processor import SiamTrackerProcessor, SiamFCBatchDataCollator
from ...common.metric_collector import SiamFCMetricCollector


def _build_siamfc_data_processor(network_data_config: dict, data_processor_config, label_generator,
                                 label_batch_data_collator):
    assert network_data_config['imagenet_normalization'] is True
    return SiamTrackerProcessor(network_data_config['template_size'], network_data_config['search_size'],
                                data_processor_config['area_factor']['template'],
                                data_processor_config['area_factor']['search'],
                                data_processor_config['augmentation']['scale_jitter_factor']['template'],
                                data_processor_config['augmentation']['scale_jitter_factor']['search'],
                                data_processor_config['augmentation']['translation_jitter_factor']['template'],
                                data_processor_config['augmentation']['translation_jitter_factor']['search'],
                                data_processor_config['augmentation']['gray_scale_probability'],
                                data_processor_config['augmentation']['color_jitter'],
                                label_generator,
                                network_data_config['interpolation_mode']), \
           SiamFCBatchDataCollator(label_batch_data_collator)


def get_additional_metric_collector():
    return SiamFCMetricCollector


def build_siamfc_tracker_data_processor(data_config, network_config: dict):
    processor_config = data_config['processor']
    if processor_config['type'] == 'SiamFC':
        label_generator, label_batch_collator = build_label_generator_and_batch_collator(network_config)
        data_processor, data_batch_collator = _build_siamfc_data_processor(
            network_config['data'], processor_config, label_generator, label_batch_collator)
    else:
        raise NotImplementedError(processor_config['type'])

    metric_collector = get_additional_metric_collector()

    return data_processor, data_batch_collator, metric_collector
