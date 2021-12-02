from criterion.builder import build_criteria_and_loss_composer
from runners.training.common.optimization.builder import build_optimization_components
from miscellanies.torch.distributed import get_world_size
from .runner import DefaultTrainer
from core.run.event_dispatcher.register import EventRegister


def build_default_training_runner(model, runner_config, data_context, config, event_register: EventRegister):
    num_epochs = config['runs']['num_epochs']

    criterion, loss_composer = build_criteria_and_loss_composer(config, runner_config, num_epochs,
                                                                data_context['iterations_per_epoch'],
                                                                event_register)

    optimizer, lr_scheduler, lr_scheduler_is_per_iteration = build_optimization_components(model, runner_config, num_epochs,
                                                                             data_context['iterations_per_epoch'])
    lr_scheduler_per_epoch = None
    lr_scheduler_per_iteration = None
    if lr_scheduler_is_per_iteration:
        lr_scheduler_per_iteration = lr_scheduler
    else:
        lr_scheduler_per_epoch = lr_scheduler

    grad_max_norm = None
    if 'clip_max_norm' in runner_config['optimizer']:
        grad_max_norm = runner_config['optimizer']['clip_max_norm']

    iteration_step = data_context['batch_size'] * get_world_size()
    trainer = DefaultTrainer(criterion, optimizer, lr_scheduler_per_iteration, lr_scheduler_per_epoch, loss_composer,
                             grad_max_norm, iteration_step)
    event_register.register_device_changed_hook(trainer)
    event_register.register_stateful_object('default_trainer', trainer)
    return trainer
