class MetricLoggerDispatcher:
    def __init__(self):
        self.loggers = {}
        self.step = None

    def register_logger(self, name, logger):
        assert name != 'all'
        assert name not in self.loggers
        self.loggers[name] = logger

    def register_metric(self, metric):
        for logger_name, metric_definition in metric.items():
            if logger_name in self.loggers:
                self.loggers[logger_name].define_metrics(metric_definition)

    def log(self, all=None, step=None, **kwargs):
        if step is None:
            step = self.step
        if all is not None:
            for logger in self.loggers.values():
                logger.log(all, step=step)

        for logger_name in self.loggers.keys():
            if logger_name in kwargs:
                logger = self.loggers[logger_name]
                logger.log(kwargs[logger_name], step=step)

    def log_summary(self, metric_data):
        for logger in self.loggers.values():
            logger.log_summary(metric_data)

    def synchronize(self):
        for logger in self.loggers.values():
            if hasattr(logger, 'synchronize'):
                logger.synchronize()

    def set_step(self, step):
        self.step = step
