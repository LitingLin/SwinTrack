def build_iter_wise_lr_scheduler(optimizer, optimizer_config: dict, num_epochs, iterations_per_epoch: int):
    lr_scheduler_config = optimizer_config['lr_scheduler']
    lr_scheduler_type = lr_scheduler_config['type']
    total_iterations = num_epochs * iterations_per_epoch
    if lr_scheduler_type == 'MultiStepLR':
        from fvcore.common.param_scheduler import MultiStepParamScheduler
        values = lr_scheduler_config['values']
        milestones = lr_scheduler_config['milestones']
        milestones = [int(round(num_epochs * milestone)) for milestone in milestones]
        lr_scheduler = MultiStepParamScheduler(values, milestones=milestones)
    else:
        raise NotImplementedError
    if 'warmup' in lr_scheduler_config:
        warmup_config = lr_scheduler_config['warmup']
        warmup_factor = warmup_config['initial_factor']
        warmup_length = warmup_config['length']
        warmup_method = warmup_config['method']

        from .iter_wise import WarmupParamScheduler

        lr_scheduler = WarmupParamScheduler(lr_scheduler, warmup_factor, warmup_length, warmup_method)

    from .iter_wise import LRMultiplier
    lr_scheduler = LRMultiplier(optimizer, lr_scheduler, total_iterations)

    return lr_scheduler
