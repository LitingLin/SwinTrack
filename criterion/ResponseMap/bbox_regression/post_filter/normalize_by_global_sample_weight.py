from miscellanies.torch.distributed.reduce_mean import reduce_mean_


def normalize_by_global_sample_weight(losses, predicted, label, context):
    weight = context['sample_weight']

    weight = weight.sum()
    weight.clamp_(min=1.e-5)
    reduce_mean_(weight)
    return [loss / weight for loss in losses]


def build_data_filter(*_):
    return normalize_by_global_sample_weight
