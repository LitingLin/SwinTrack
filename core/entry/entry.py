from miscellanies.torch.distributed import is_main_process
from contextlib import nullcontext
import torch.distributed
from miscellanies.torch.distributed import is_dist_available_and_initialized, get_world_size
import socket
import pprint
import os
from miscellanies.yaml_ops import load_yaml
from .sweep_utils import prepare_sweep
from .mixin_utils import load_static_mixin_config_and_apply_rules
from .build_and_run import build_and_run


def update_output_dir(args):
    # redirect output path with run_id
    if args.output_dir is not None:
        args.output_dir = os.path.join(args.output_dir, args.run_id)
        os.makedirs(args.output_dir, exist_ok=True)


def entry(runtime_vars):
    config_path = os.path.join(runtime_vars.config_path, runtime_vars.method_name, runtime_vars.config_name, 'config.yaml')
    config = load_yaml(config_path)
    if runtime_vars.mixin_config is not None:
        load_static_mixin_config_and_apply_rules(runtime_vars, config)

    my_hostname = socket.gethostname()
    my_ip = socket.gethostbyname(my_hostname)
    print(f'Hostname: {my_hostname}')
    print(f'IP: {my_ip}')
    if is_dist_available_and_initialized():
        host_names = [None] * get_world_size()
        torch.distributed.all_gather_object(host_names, [my_ip, my_hostname])

        host_names = {ip: hostname for ip, hostname in host_names}
        print('Distributed Group:')
        pprint.pprint(host_names)
    else:
        host_names = {my_ip: my_hostname}

    if not runtime_vars.do_sweep:
        update_output_dir(runtime_vars)
    wandb_instance = None
    if runtime_vars.wandb_distributed_aware or not is_dist_available_and_initialized():
        from .setup_wandb import setup_wandb
        wandb_instance = setup_wandb(runtime_vars, config, str(host_names))
        if runtime_vars.do_sweep:
            runtime_vars.run_id = wandb_instance.id
            update_output_dir(runtime_vars)
    else:
        if is_main_process():
            from .setup_wandb import setup_wandb
            wandb_instance = setup_wandb(runtime_vars, config, str(host_names))

        if runtime_vars.do_sweep:
            if is_main_process():
                run_id = [wandb_instance.id]
            else:
                run_id = [None]
            torch.distributed.broadcast_object_list(run_id)
            runtime_vars.run_id = run_id[0]
            update_output_dir(runtime_vars)

    wandb_context = wandb_instance if wandb_instance is not None else nullcontext()
    with wandb_context:
        if runtime_vars.do_sweep:
            prepare_sweep(runtime_vars, wandb_instance, config)
        build_and_run(runtime_vars, config, wandb_instance)
