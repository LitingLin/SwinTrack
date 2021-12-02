import collections.abc
import importlib
from core.run.event_dispatcher.register import EventRegister
_this_module_prefix = 'criterion'


def _build_filter(filter_parameters, name, module_path_prefix, config):
    filter_type = filter_parameters['type']
    filter_module = module_path_prefix + f'.{name}.' + filter_type
    filter_build_function = getattr(importlib.import_module(filter_module), 'build_data_filter')
    filter_function = filter_build_function(filter_parameters, config)
    return filter_function


def _get_filter(parameters, name, module_path_prefix, config):
    if name in parameters:
        filter_functions = []
        filter_parameters = parameters[name]
        if isinstance(filter_parameters, collections.abc.Sequence):
            for current_filter_parameters in filter_parameters:
                filter_functions.append(_build_filter(current_filter_parameters, name, module_path_prefix, config))
        else:
            filter_functions.append(_build_filter(filter_parameters, name, module_path_prefix, config))
    else:
        filter_functions = None
    return filter_functions


def build_single_scale(loss_parameters, module_path_prefix, head_output_protocol, network_config, index_of_scale):
    protocol_module_path_prefix = module_path_prefix + f'.{head_output_protocol}'
    loss_modules = []

    global_data_pre_filter = _get_filter(loss_parameters, 'pre_filter', protocol_module_path_prefix, network_config)

    branches_loss_parameters = loss_parameters['branches']

    for branch_name, branch_parameters in branches_loss_parameters.items():
        branch_module_prefix = protocol_module_path_prefix + '.' + branch_name

        loss_functions = []
        loss_data_adaptor_functions = []
        loss_reduction_functions = []

        criteria_parameters = branch_parameters['criterion']
        for criterion_name, criterion_parameters in criteria_parameters.items():
            module = importlib.import_module(branch_module_prefix + '.' + criterion_name)
            loss_function_build_function = getattr(module, 'build_' + criterion_name)
            loss_function, loss_data_adaptor_function, loss_reduction_function = loss_function_build_function(criterion_parameters, network_config, index_of_scale)
            loss_functions.append(loss_function)
            loss_data_adaptor_functions.append(loss_data_adaptor_function)
            loss_reduction_functions.append(loss_reduction_function)

        pre_sample_filter_function = _get_filter(branch_parameters, 'pre_filter', branch_module_prefix, network_config)
        post_sample_filter_function = _get_filter(branch_parameters, 'post_filter', branch_module_prefix, network_config)

        loss_modules.append((branch_name, pre_sample_filter_function, loss_functions, loss_data_adaptor_functions, loss_reduction_functions, post_sample_filter_function))

    from .single_scale_criterion import SingleScaleCriterion
    criterion = SingleScaleCriterion(global_data_pre_filter, loss_modules)

    return criterion


def build_multi_scale(loss_parameters, module_path_prefix, head_output_protocol, config):
    multi_scales_parameters = loss_parameters['multi_scale']

    multi_scale_module_prefix = module_path_prefix + '.multi_scale'
    global_data_pre_filter = _get_filter(multi_scales_parameters, 'pre_filter', multi_scale_module_prefix, config)

    scale_dispatcher_parameters = multi_scales_parameters['dispatcher']
    dispatcher_function_type = scale_dispatcher_parameters['type']
    dispatcher_module = importlib.import_module(multi_scale_module_prefix + '.dispatcher.' + dispatcher_function_type)
    dispatcher_build_function = getattr(dispatcher_module, 'build_multi_scale_data_dispatcher')
    multi_scale_data_dispatcher = dispatcher_build_function(scale_dispatcher_parameters, config)

    scales_parameters = multi_scales_parameters['scales']
    single_scale_criterion = [build_single_scale(scale_parameters, module_path_prefix, head_output_protocol, config, i) for i, scale_parameters in enumerate(scales_parameters)]

    from .multi_scale_criterion import MultiScaleCriterion
    return MultiScaleCriterion(global_data_pre_filter, multi_scale_data_dispatcher, single_scale_criterion)


def build_criteria_and_loss_composer(network_config, optimizer_config, num_epochs, iterations_per_epoch, event_register: EventRegister):
    loss_parameters = optimizer_config['loss']
    head_output_protocol = network_config['head']['output_protocol']['type']
    if 'multi_scale' in loss_parameters:
        loss_module = build_multi_scale(loss_parameters, _this_module_prefix, head_output_protocol, network_config)
    else:
        loss_module = build_single_scale(loss_parameters, _this_module_prefix, head_output_protocol, network_config, None)
    from .composer.builder import build_loss_composer
    loss_composer = build_loss_composer(optimizer_config, num_epochs, iterations_per_epoch, event_register)
    return loss_module, loss_composer
