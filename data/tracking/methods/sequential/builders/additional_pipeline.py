from core.run.event_dispatcher.register import EventRegister
from data.tracking.post_processor.builder import build_post_processor
from data.tracking.post_processor.bounding_box.default import build_bounding_box_post_processor
from .components.host_cache import build_tracking_procedure_template_cache
from .components.curation_parameter_provider import build_curation_parameter_provider
from .components.metric_collector import build_metric_collector
from ..pipeline.hook.sequence_collection import SequenceTrackingPerformanceMetricCollector
from ..pipeline.evaluation_host import EvaluationHostPipeline


def build_sequential_sampling_additional_data_pipelines(runtime_vars, branch_config, data_config, runner_config, config, event_register: EventRegister, data_source_context, capable_for_training):
    assert data_config['type'] == 'Sequential'
    tracking_data_config = data_config['tracking']
    tracking_evaluation_config = branch_config['tracking']
    enable_tracking_performance_metrics = branch_config['metrics']['enabled']

    network_data_config = config['data']

    batch_size = data_source_context['batch_size']

    host_processor_hook = None
    if enable_tracking_performance_metrics:
        host_processor_hook = SequenceTrackingPerformanceMetricCollector()

    bounding_box_post_processor = build_bounding_box_post_processor(config)

    template_feature_cache_service, template_image_mean_cache_service = build_tracking_procedure_template_cache(runtime_vars, branch_config, batch_size)

    data_pipeline = {'data_pipeline': []}

    post_processor = build_post_processor(config, tracking_evaluation_config)

    if runner_config['type'] == 'default_evaluation':
        assert tracking_evaluation_config['type'] == 'SiamFC' and tracking_data_config['type'] == 'SiamFC'
        curation_parameter_provider = build_curation_parameter_provider(tracking_evaluation_config,
                                                                        tracking_data_config['pre_processing']['search']['area_factor'])

        host_post_processor = EvaluationHostPipeline(batch_size,
                                                     network_data_config['template_size'], network_data_config['search_size'],
                                                     curation_parameter_provider,
                                                     template_feature_cache_service, template_image_mean_cache_service,
                                                     host_processor_hook, post_processor, bounding_box_post_processor,
                                                     network_data_config['interpolation_mode'])
        data_pipeline['data_pipeline'].append(host_post_processor)
        data_pipeline['tracker_evaluator'] = host_post_processor
    else:
        raise NotImplementedError(runner_config['type'])
    event_register.register_device_changed_hook(host_post_processor)
    event_register.register_epoch_begin_hook(host_post_processor)
    event_register.register_epoch_end_hook(host_post_processor)

    if enable_tracking_performance_metrics:
        metric_collector = build_metric_collector(runtime_vars, branch_config, data_source_context, event_register, not capable_for_training)
        data_pipeline['data_pipeline'].insert(0, metric_collector)
    return data_pipeline
