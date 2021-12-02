import wandb
import copy
from core.run.event_dispatcher.register import EventRegister


class WandbLogger:
    def __init__(self, wandb_instance: wandb.wandb_sdk.wandb_run.Run, logging_frequency: int, prefix=None,
                 with_epoch=False):
        self.instance = wandb_instance
        self.logging_frequency = logging_frequency
        self.prefix = prefix
        self.with_epoch = with_epoch

    def log_summary(self, metrics):
        self.instance.summary.update(metrics)

    def on_epoch_begin(self, epoch):
        self.logging_step = 0
        self.epoch = epoch
        self.metrics = None
        self.step = None

    def on_epoch_end(self, _):
        if self.metrics is not None:
            self.instance.log(self.metrics, step=self.step)

        self.metrics = None
        self.step = None

    def log(self, metrics, step):
        if self.prefix is not None:
            metrics = {self.prefix + k: v for k, v in metrics.items()}

        if self.with_epoch:
            metrics['epoch'] = self.epoch

        if self.logging_step % self.logging_frequency == 0:
            self.instance.log(metrics, step=step)
            self.metrics = None
            self.step = None
        else:
            self.metrics = metrics
            self.step = step

        self.logging_step += 1

    def define_metrics(self, metric_definitions):
        for metric_definition in metric_definitions:
            if self.prefix is not None:
                metric_definition = copy.copy(metric_definition)
                metric_definition['name'] = metric_definition['name'] + self.prefix

            self.instance.define_metric(**metric_definition)


class WandbEpochSummaryLogger:
    def __init__(self, wandb_instance: wandb.wandb_sdk.wandb_run.Run, summary_method, prefix=None, with_epoch=False):
        assert summary_method == 'mean'
        self.instance = wandb_instance
        self.prefix = prefix
        self.with_epoch = with_epoch

    def log_summary(self, metrics):
        self.instance.summary.update(metrics)

    def on_epoch_begin(self, epoch):
        self.epoch = epoch
        self.metrics = {}

    def on_epoch_end(self, epoch):
        if len(self.metrics) > 0:
            epoch_metrics = {}
            for metric_name, (metric_total, metric_count) in self.metrics.items():
                epoch_metrics[metric_name] = metric_total / metric_count
            if self.with_epoch:
                epoch_metrics['epoch'] = epoch
            self.instance.log(epoch_metrics, step=self.step)

    def log(self, metrics, step):
        if self.prefix is not None:
            metrics = {self.prefix + k: v for k, v in metrics.items()}

        for metric_name, metric_value in metrics.items():
            if metric_name not in self.metrics:
                self.metrics[metric_name] = [metric_value, 1]
            else:
                epoch_metric_statistic = self.metrics[metric_name]
                epoch_metric_statistic[0] += metric_value
                epoch_metric_statistic[1] += 1

        self.step = step

    def define_metrics(self, metric_definitions):
        for metric_definition in metric_definitions:
            if self.prefix is not None:
                metric_definition = copy.copy(metric_definition)
                metric_definition['name'] = metric_definition['name'] + self.prefix

            self.instance.define_metric(**metric_definition)


def _wandb_logger_register_event_callback(logger, event_register: EventRegister):
    event_register.register_epoch_begin_hook(logger)
    event_register.register_epoch_end_hook(logger)


def build_wandb_logger(logging_config, wandb_instance, branch_event_register: EventRegister):
    if wandb_instance is None:
        return []
    if logging_config is None:
        logger = WandbLogger(wandb_instance, 1)
        _wandb_logger_register_event_callback(logger, branch_event_register)
        return [logger]
    enable_per_iteration_logging = True
    enable_per_epoch_logging = False
    per_iteration_logging_prefix = None
    per_epoch_logging_prefix = None
    with_epoch = False
    logging_interval = 1
    per_epoch_logging_summary_method = 'mean'

    if 'wandb' in logging_config and 'interval' in logging_config['wandb']:
        logging_interval = logging_config['wandb']['interval']
    elif 'interval' in logging_config:
        logging_interval = logging_config['interval']

    if 'metric_prefix' in logging_config:
        per_iteration_logging_prefix = logging_config['metric_prefix']
        per_epoch_logging_prefix = logging_config['metric_prefix']
    if 'wandb' in logging_config:
        wandb_config = logging_config['wandb']
        if 'with_epoch' in wandb_config:
            with_epoch = wandb_config['with_epoch']

        if 'per_iteration_logging' in wandb_config:
            per_iteration_logging_config = wandb_config['per_iteration_logging']
            enable_per_iteration_logging = per_iteration_logging_config['enabled']
            if 'prefix' in per_iteration_logging_config:
                per_iteration_logging_prefix = per_iteration_logging_prefix + per_iteration_logging_config['prefix'] if per_iteration_logging_prefix is not None else per_iteration_logging_config['prefix']

        if 'per_epoch_logging' in wandb_config:
            per_epoch_logging_config = wandb_config['per_epoch_logging']
            enable_per_epoch_logging = per_epoch_logging_config['enabled']
            if 'prefix' in per_epoch_logging_config:
                per_epoch_logging_prefix = per_epoch_logging_prefix + per_epoch_logging_config['prefix'] if per_epoch_logging_prefix is not None else per_epoch_logging_config['prefix']

            if 'summary_method' in per_epoch_logging_config:
                per_epoch_logging_summary_method = per_epoch_logging_config['summary_method']

    loggers = []
    if enable_per_iteration_logging:
        logger = WandbLogger(wandb_instance, logging_interval, per_iteration_logging_prefix, with_epoch)
        _wandb_logger_register_event_callback(logger, branch_event_register)
        loggers.append(logger)
    if enable_per_epoch_logging:
        logger = WandbEpochSummaryLogger(wandb_instance, per_epoch_logging_summary_method, per_epoch_logging_prefix, with_epoch)
        _wandb_logger_register_event_callback(logger, branch_event_register)
        loggers.append(logger)
    return loggers
