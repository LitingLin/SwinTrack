from miscellanies.torch.distributed import is_dist_available_and_initialized, get_world_size
import torch.distributed


def reduce_mean_(tensor):
    if is_dist_available_and_initialized():
        torch.distributed.all_reduce(tensor)
        tensor.div_(get_world_size())
