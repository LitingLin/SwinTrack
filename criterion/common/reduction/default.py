def loss_mean_reduction_function(loss, *_):
    return loss.mean()


def loss_sum_reduction_function(loss, *_):
    return loss.sum()


def loss_reduce_by_weight(loss, pred, label, context):
    return (loss * context['sample_weight']).sum()


def build_loss_reduction_function(loss_parameters: dict):
    if 'reduce' not in loss_parameters:
        loss_reduction_function = loss_mean_reduction_function
    else:
        loss_reduction_function_parameters = loss_parameters['reduce']
        if loss_reduction_function_parameters == 'mean':
            loss_reduction_function = loss_mean_reduction_function
        elif loss_reduction_function_parameters == 'sum':
            loss_reduction_function = loss_sum_reduction_function
        elif loss_reduction_function_parameters == 'weighted':
            loss_reduction_function = loss_reduce_by_weight
        else:
            loss_reduction_function = None

    return loss_reduction_function
