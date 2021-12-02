from .logger import MetricLoggerDispatcher
from typing import Optional


_logger: Optional[MetricLoggerDispatcher] = None


class enable_logger:
    def __init__(self, logger: MetricLoggerDispatcher):
        global _logger
        self.last_logger = _logger
        _logger = logger

    def __enter__(self):
        pass

    def __exit__(self, exc_type, exc_val, exc_tb):
        global _logger
        _logger = self.last_logger


def disable_logger():
    global _logger
    _logger = None


def get_logger():
    return _logger
