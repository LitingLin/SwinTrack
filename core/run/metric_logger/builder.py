from .logger import MetricLoggerDispatcher
from ._local import build_local_logger
from ._wandb import build_wandb_logger
from core.run.event_dispatcher.register import EventRegister


def build_logger(logging_config, wandb_instance, branch_event_register: EventRegister, global_event_register: EventRegister):
    local_logger = build_local_logger(logging_config, branch_event_register, global_event_register)
    wandb_logger = build_wandb_logger(logging_config, wandb_instance, branch_event_register)
    logger_dispatcher = MetricLoggerDispatcher()
    logger_dispatcher.register_logger('local', local_logger)
    for sub_wandb_logger in wandb_logger:
        logger_dispatcher.register_logger('wandb', sub_wandb_logger)
    return logger_dispatcher
