import math
from miscellanies.torch.distributed import is_dist_available_and_initialized, is_main_process
import torch.distributed
from miscellanies.yaml_ops import load_yaml
import os
from .mixin_utils import apply_mixin_rules


def _update_sweep_config_(config: dict):
    wandb_tuner_config = config['tune']
    for parameter in wandb_tuner_config['parameters'].values():
        if 'distribution' in parameter and parameter['distribution'] in ('log_uniform', 'q_log_uniform', 'log_normal', 'q_log_normal'):
            if 'min_raw' in parameter:
                parameter['min'] = math.log(parameter['min_raw'])
                del parameter['min_raw']
            if 'max_raw' in parameter:
                parameter['max'] = math.log(parameter['max_raw'])
                del parameter['max_raw']
    return config


def get_sweep_config(args):
    if args.sweep_config is not None:
        if args.sweep_config.startswith('/' or '\\'):
            config_path = os.path.join(args.config_path, args.sweep_config)
        else:
            config_path = os.path.join(args.config_path, args.method_name, args.config_name, 'sweep', args.sweep_config)
    else:
        config_path = os.path.join(args.config_path, args.method_name, args.config_name, 'sweep', 'sweep.yaml')
    config = load_yaml(config_path)
    _update_sweep_config_(config)
    return config


def prepare_sweep(args, wandb_instance, config):
    # get the config of this run from wandb server
    if is_main_process():
        this_run_config = wandb_instance.config.as_dict()
    else:
        this_run_config = None
    if is_dist_available_and_initialized():
        object_list = [this_run_config]
        torch.distributed.broadcast_object_list(object_list, src=0)
        this_run_config, = object_list
    sweep_config = get_sweep_config(args)
    apply_mixin_rules(sweep_config['mixin'], config, this_run_config)
    if args.debug:
        import pprint
        pprint.pprint(config)
