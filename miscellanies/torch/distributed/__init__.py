import torch.distributed as dist
import os
import torch
import sys


def is_dist_available_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_available_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    if not is_dist_available_and_initialized():
        return 0
    return dist.get_rank()


def get_backend():
    if not is_dist_available_and_initialized():
        return None
    return dist.get_backend()


def is_main_process():
    return get_rank() == 0


def setup_for_distributed(is_node_master):
    """
    This function disables printing when not in master process
    # """
    if not is_node_master:
        f = open(os.devnull, 'w')
        sys.stdout = f


def init_distributed_mode(args):
    if 'RANK' not in os.environ:
        print('Not using distributed mode')
        args.distributed = False
        args.rank = None
        args.world_size = None
        args.local_rank = None
        args.local_world_size = None
        return

    args.distributed = True
    args.rank = int(os.environ["RANK"])
    args.world_size = int(os.environ['WORLD_SIZE'])
    args.local_rank = int(os.environ['LOCAL_RANK'])
    args.local_world_size = int(os.environ['LOCAL_WORLD_SIZE'])

    if 'cuda' in args.device:
        torch.cuda.set_device(args.local_rank)
        args.dist_backend = 'nccl'
    else:
        args.dist_backend = 'gloo'
    print(f'| distributed init (rank {args.rank}[{args.rank // args.local_world_size}.{args.local_rank}]/{args.world_size}) using {args.dist_backend}', flush=True)
    torch.distributed.init_process_group(backend=args.dist_backend, init_method='env://',
                                         world_size=args.world_size, rank=args.rank)
    torch.distributed.barrier()
    if not args.debug:
        setup_for_distributed(args.local_rank == 0)
