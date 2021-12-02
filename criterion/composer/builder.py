from core.run.event_dispatcher.register import EventRegister


def build_single_scale_loss_composer(loss_parameters, total_iterations):
    from .composer import LinearWeightScheduler, ConstantWeightScheduler, LossComposer
    weight_schedulers = []
    display_names = []
    display_prefix = None
    if 'display_prefix' in loss_parameters:
        display_prefix = loss_parameters['display_prefix']
    display_postfix = None
    if 'display_postfix' in loss_parameters:
        display_postfix = loss_parameters['display_postfix']
    branches_loss_parameters = loss_parameters['branches']
    for branches_loss_parameter in branches_loss_parameters.values():
        branch_criterion_parameters = branches_loss_parameter['criterion']
        for loss_name, loss_parameter in branch_criterion_parameters.items():
            if isinstance(loss_parameter['weight'], dict):
                weight_parameters = loss_parameter['weight']
                assert weight_parameters['scheduler'] == 'linear'
                weight_scheduler = LinearWeightScheduler(weight_parameters['initial_value'],
                                                         weight_parameters['ultimate_value'],
                                                         0, int(round(weight_parameters['length'] * total_iterations)),
                                                         weight_parameters['per_iteration'])
            elif isinstance(loss_parameter['weight'], (int, float)):
                weight_scheduler = ConstantWeightScheduler(loss_parameter['weight'])
            else:
                raise NotImplementedError
            weight_schedulers.append(weight_scheduler)
            display_names.append(loss_parameter['display_name'])
    return LossComposer(weight_schedulers, display_names, display_prefix, display_postfix)


def build_multi_scale_loss_composer(multi_scale_loss_parameters, loss_composers):
    reduction_method = multi_scale_loss_parameters['reduce']
    if reduction_method == 'sum':
        from .multi_scale import _sum_compose
        reduce_fn = _sum_compose
    else:
        raise NotImplementedError(f'Unknown parameter {reduction_method}')
    from .multi_scale import MultiScaleLossComposer
    return MultiScaleLossComposer(reduce_fn, loss_composers)


def build_loss_composer(optimizer_config: dict, num_epochs, iterations_per_epoch: int, event_register: EventRegister):
    loss_parameters = optimizer_config['loss']
    total_iterations = num_epochs * iterations_per_epoch
    if 'multi_scale' in loss_parameters:
        multi_scale_parameters = loss_parameters['multi_scale']
        scales_parameters = multi_scale_parameters['scales']
        single_scale_loss_composers = [build_single_scale_loss_composer(scale_parameters, total_iterations) for scale_parameters in scales_parameters]
        loss_composer = build_multi_scale_loss_composer(multi_scale_parameters, single_scale_loss_composers)
    else:
        loss_composer = build_single_scale_loss_composer(loss_parameters, total_iterations)

    event_register.register_stateful_object('loss_composer', loss_composer)
    event_register.register_iteration_end_hook(loss_composer)
    event_register.register_epoch_begin_hook(loss_composer)

    return loss_composer
