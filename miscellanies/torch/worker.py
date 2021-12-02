import torch.utils.data


def is_in_worker_process():
    return torch.utils.data.get_worker_info() is not None
