from miscellanies.torch.distributed import init_distributed_mode
import numpy as np
import argparse
import os
from .train_args import get_train_args_parser
from .run_id import generate_run_id
from .entry import entry
from core.workaround.numpy import numpy_no_multithreading


def spawn_workers(args):
    # launch pytorch distributed data parallel(DDP) workers
    import sys
    command_args = sys.argv

    deterministic_rng = np.random.Generator(np.random.PCG64(args.conf_seed))
    rdzv_port = deterministic_rng.integers(10000, 50000)

    train_script_path = command_args[0]
    torch_run_args = ['--nnodes', str(args.distributed_nnodes), '--nproc_per_node', str(args.distributed_nproc_per_node),
                      '--rdzv_endpoint', f'{args.master_address}:{rdzv_port}', '--max_restarts', str(0),
                      '--rdzv_id', args.run_id, '--node_rank', str(args.distributed_node_rank),
                      '--rdzv_backend', 'static', '--master_addr', args.master_address, '--master_port', str(rdzv_port),
                      train_script_path]

    index_of_arg = 1
    while index_of_arg < len(command_args):
        command_arg = command_args[index_of_arg]
        if command_arg in ('--distributed_nnodes', '--distributed_nproc_per_node', '--conf_seed', '--run_id', '--distributed_node_rank'):
            index_of_arg += 2
        elif command_arg in ('--distributed_do_spawn_workers', '--kill_other_python_processes'):
            index_of_arg += 1
        else:
            torch_run_args.append(command_arg)
            index_of_arg += 1
    torch_run_args.extend(
        ['--run_id', args.run_id,
         '--conf_seed', str(deterministic_rng.integers(10000000))])
    print(f'Executing torch.distributed.run.main({torch_run_args})')
    import torch.distributed.run
    return torch.distributed.run.main(torch_run_args)


def _kill_other_python_processes():
    # Commonly used in hyper-parameter search, prevent interference from the old run.
    # Be careful with this since it will kill all python processes other than the current one and its parent(typically the hyper-parameter tunning host process).
    self_pid = os.getpid()
    parent_pid = os.getppid()
    import psutil
    for proc in psutil.process_iter():
        try:
            process_name = proc.name()
            if process_name == 'python':
                pid = proc.pid
                if pid == self_pid or pid == parent_pid:
                    continue
                else:
                    proc.kill()
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            pass


def _remove_ddp_parameter(args):
    # remove ddp related parameters from args
    del args.distributed_node_rank
    del args.distributed_nnodes
    del args.distributed_nproc_per_node
    del args.distributed_do_spawn_workers


def setup_arg_parser():
    parser = argparse.ArgumentParser('Model training & evaluation entry script', parents=[get_train_args_parser()])
    parser.add_argument('--watch_model_parameters', action='store_true',
                        help='watch the parameters of model using wandb')
    parser.add_argument('--watch_model_gradients', action='store_true',
                        help='watch the gradients of model using wandb')
    parser.add_argument('--watch_model_freq', default=1000, type=int,
                        help='model watching frequency')

    parser.add_argument('--weight_path', type=str, help='path to the .pth weight file')
    return parser


def main(root_path):
    numpy_no_multithreading()
    parser = setup_arg_parser()
    args = parser.parse_args()

    args.root_path = root_path
    args.config_path = os.path.join(root_path, 'config')

    if args.kill_other_python_processes:
        _kill_other_python_processes()

    if args.run_id is None:
        args.run_id = generate_run_id(args)

    if args.distributed_do_spawn_workers:
        return spawn_workers(args)

    _remove_ddp_parameter(args)
    init_distributed_mode(args)

    entry(args)
