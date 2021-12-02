import torch.nn as nn


def _compute_loss(pred, label, input_data_filter, loss_functions, loss_data_adaptors, loss_reduction_functions, output_data_filter):
    context = {}
    if input_data_filter is not None:
        for filter_ in input_data_filter:
            pred, label = filter_(pred, label, context)
    losses = []
    for loss_fn, loss_data_adaptor, loss_reduction_function in zip(loss_functions, loss_data_adaptors, loss_reduction_functions):
        ok, loss_fn_inputs = loss_data_adaptor(pred, label, context)
        if not ok:
            loss = loss_fn_inputs
        else:
            if isinstance(loss_fn_inputs, (list, tuple)):
                loss = loss_fn(*loss_fn_inputs)
            elif isinstance(loss_fn_inputs, dict):
                loss = loss_fn(**loss_fn_inputs)
            else:
                loss = loss_fn(loss_fn_inputs)
            loss = loss_reduction_function(loss, pred, label, context)
        losses.append(loss)
    if output_data_filter is not None:
        for filter_ in output_data_filter:
            losses = filter_(losses, pred, label, context)
    return losses


class SingleScaleCriterion(nn.Module):
    def __init__(self, global_data_filter, loss_modules):
        super(SingleScaleCriterion, self).__init__()
        self.global_data_filter = global_data_filter
        for module_name, module_data_pre_filter, module_loss_functions, loss_data_adaptors, loss_reduction_functions, module_data_post_filter in loss_modules:
            self.__setattr__(module_name, nn.ModuleList(module_loss_functions))
        self.loss_modules = loss_modules

    def forward(self, pred, label):
        if self.global_data_filter is not None:
            for filter_ in self.global_data_filter:
                pred, label = filter_(pred, label)
        losses = []
        for module_name, module_data_pre_filter, _, loss_data_adaptors, loss_reduction_functions, module_data_post_filter in self.loss_modules:
            module_loss_functions = self.__getattr__(module_name)
            losses.extend(_compute_loss(pred, label, module_data_pre_filter, module_loss_functions, loss_data_adaptors, loss_reduction_functions, module_data_post_filter))
        return losses
