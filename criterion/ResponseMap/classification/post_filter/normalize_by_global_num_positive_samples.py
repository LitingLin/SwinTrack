from miscellanies.torch.distributed.reduce_mean import reduce_mean_


def reweight_by_num_pos(losses, pred, label, _):
    num_pos = label['num_positive_samples']
    reduce_mean_(num_pos)
    num_pos = max(num_pos.item(), 1e-4)

    return [loss / num_pos for loss in losses]


def build_data_filter(*_):
    return reweight_by_num_pos
