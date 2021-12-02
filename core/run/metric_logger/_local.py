from miscellanies.torch.metric_logger import MetricLogger, SmoothedValue
import copy
from core.run.event_dispatcher.register import EventRegister


def _load_metric_definitions(metric_definitions, logger: MetricLogger):
    name_check = set()
    for metric_definition in metric_definitions:
        window_size = 20
        fmt_string = "{median:.4f} ({global_avg:.4f})"
        name = metric_definition['name']
        assert name not in name_check
        name_check.add(name)
        if 'window_size' in metric_definition:
            window_size = metric_definition['window_size']
        if 'fmt' in metric_definition:
            fmt_string = metric_definition['fmt']
        logger.add_meter(name, SmoothedValue(window_size, fmt_string))


class LocalLoggerWrapper:
    def __init__(self, print_freq, prefix, print_epoch_average, header):
        self.print_freq = print_freq
        self.prefix = prefix
        self.metric_definitions = []
        self.print_epoch_average = print_epoch_average
        self.header = header
        self.metric_logger = None
        self.summary_metrics = None

    def on_epoch_begin(self, epoch):
        self.metric_logger = MetricLogger(delimiter=' ')
        self.epoch_header = self.header.format(epoch=epoch)
        self.summary_metrics = None
        _load_metric_definitions(self.metric_definitions, self.metric_logger)

    def on_epoch_end(self, epoch):
        if self.print_epoch_average:
            print("Averaged stats:", self.metric_logger)
        if self.summary_metrics is not None:
            print(f'Epoch [{epoch}] summary metrics:\n' + ('\n'.join("{}: {}".format(k, v) for k, v in self.summary_metrics.items())))
        self.summary_metrics = None
        self.metric_logger = None
        self.epoch_header = None

    def on_finished(self):
        if self.summary_metrics is not None:
            print('summary metrics:\n' + ('\n'.join("{}: {}".format(k, v) for k, v in self.summary_metrics.items())))
            self.summary_metrics = None

    def log_every(self, iterable):
        return self.metric_logger.log_every(iterable, self.print_freq, self.epoch_header)

    def log(self, metrics, step):
        if self.prefix is not None:
            metrics = {self.prefix + k: v for k, v in metrics.items()}
        self.metric_logger.update(**metrics)

    def define_metrics(self, metric_definitions):
        if self.prefix is not None:
            metric_definitions = copy.copy(metric_definitions)
            for metric_definition in metric_definitions:
                metric_definition['name'] = self.prefix + metric_definition['name']
        self.metric_definitions.extend(metric_definitions)
        if self.metric_logger is not None:
            _load_metric_definitions(metric_definitions, self.metric_logger)

    def synchronize(self):
        self.metric_logger.synchronize_between_processes()

    def log_summary(self, summary_metrics):
        if self.summary_metrics is None:
            self.summary_metrics = {}
        self.summary_metrics.update(summary_metrics)


def _local_logger_register_event_callback(logger, branch_event_register: EventRegister, global_event_register: EventRegister):
    branch_event_register.register_epoch_begin_hook(logger)
    branch_event_register.register_epoch_end_hook(logger)
    global_event_register.register_finished_hook(logger)


def build_local_logger(logging_config, branch_event_register: EventRegister, global_event_register: EventRegister):
    if logging_config is None:
        logger = LocalLoggerWrapper(1, None, False, '')
        _local_logger_register_event_callback(logger, branch_event_register, global_event_register)
        return logger

    interval = 1
    if 'local' in logging_config and 'interval' in logging_config['local']:
        interval = logging_config['local']['interval']
    elif 'interval' in logging_config:
        interval = logging_config['interval']

    header = ''
    prefix = None
    if 'metric_prefix' in logging_config:
        prefix = logging_config['metric_prefix']
    print_epoch_average = False
    if 'local' in logging_config:
        local_logger_config = logging_config['local']
        if 'epoch_summary' in local_logger_config:
            print_epoch_average = local_logger_config['epoch_summary']['enabled']
            assert local_logger_config['epoch_summary']['method'] == 'mean'
        if 'header' in local_logger_config:
            header = local_logger_config['header']

    logger = LocalLoggerWrapper(interval, prefix, print_epoch_average, header)
    _local_logger_register_event_callback(logger, branch_event_register, global_event_register)
    return logger
