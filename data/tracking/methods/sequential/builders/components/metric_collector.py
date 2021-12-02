import os
from ...metric_collector.metric_collector import EvaluationMetricsCollector
from ...metric_collector.on_the_fly_constructor import _OnTheFlyConstructor
from core.run.event_dispatcher.register import EventRegister


def build_metric_collector(runtime_vars, branch_config, data_source_context, event_register: EventRegister, has_training_run):
    multiple_run = not (has_training_run or branch_config['epoch_interval'] < 0)
    summary_by_mean = (not has_training_run) and branch_config['epoch_interval'] > 0

    output_dir = runtime_vars.output_dir

    if output_dir is not None:
        output_dir = os.path.join(output_dir, branch_config['metrics']['output_path'])

    evaluation_metrics_collector = EvaluationMetricsCollector(runtime_vars.config_name, data_source_context['dataset_sequence_names'],
                                                              _OnTheFlyConstructor(branch_config['metrics']['handler']),
                                                              multiple_run, output_dir, summary_by_mean)
    event_register.register_started_hook(evaluation_metrics_collector)
    event_register.register_finished_hook(evaluation_metrics_collector)
    event_register.register_epoch_begin_hook(evaluation_metrics_collector)
    event_register.register_epoch_end_hook(evaluation_metrics_collector)
    return evaluation_metrics_collector
