def build_warmup_scheduler(warmup_config, ultimate_value, iterations_per_epoch, epochs):
    per_iteration = warmup_config['per_iteration']
    method = warmup_config['method']
    length_ratio = warmup_config['length']
    total_iterations = iterations_per_epoch * epochs
    assert 0 <= warmup_config['length'] <= 1
    if per_iteration:
        warmup_steps = int(round(total_iterations * length_ratio))
    else:
        warmup_steps = int(round(epochs * length_ratio))
    warmup_value = ultimate_value * warmup_config['initial_factor']

    if method == 'constant':
        from .scheduler.constant_warmup_scheduler import ConstantWarmupScheduler
        scheduler = ConstantWarmupScheduler(warmup_steps, warmup_value, ultimate_value, per_iteration)
    elif method == 'linear':
        from .scheduler.linear_scheduler import LinearScheduler
        scheduler = LinearScheduler(warmup_value, ultimate_value, 0, warmup_steps, per_iteration)
    else:
        raise NotImplementedError(method)
    return scheduler
