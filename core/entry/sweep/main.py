import wandb
import os
from miscellanies.yaml_ops import load_yaml
from ..sweep_utils import get_sweep_config
from ..run_id import generate_run_id


def setup_sweep_argparse():
    import argparse
    arg_parser = argparse.ArgumentParser(description='Hyperparameter tuning script')
    arg_parser.add_argument('method_name', type=str)
    arg_parser.add_argument('config_name', type=str)
    arg_parser.add_argument('--sweep_config', type=str)
    arg_parser.add_argument('--sweep_id', type=str)
    arg_parser.add_argument('--agents_run_limit', type=int)
    arg_parser.add_argument('--run_id', type=str)
    arg_parser.add_argument('--output_dir', type=str)
    arg_parser.add_argument('--mixin_config', type=str, action='append')
    return arg_parser


def _get_program_command(args, unknown_args):
    train_script = os.path.join(args.root_path, 'main.py')
    command = ['python', train_script, args.method_name, args.config_name,
               *unknown_args, '--do_sweep', '--kill_other_python_processes']

    if args.mixin_config is not None:
        for mixin_config in args.mixin_config:
            command += ['--mixin_config', mixin_config]

    if args.sweep_config is not None:
        command += ['--sweep_config', args.sweep_config]
    if args.output_dir:
        command += ['--output_dir', args.output_dir]
    return train_script, command


def setup_sweep(args, unknown_args, project_name):
    sweep_config = get_sweep_config(args)['tune']
    sweep_config['program'], sweep_config['command'] = _get_program_command(args, unknown_args)

    sweep_id = wandb.sweep(sweep_config, project=project_name)
    args.sweep_id = sweep_id


def sweep_main(root_path):
    args, unknown_args = setup_sweep_argparse().parse_known_args()
    config_path = os.path.join(root_path, 'config')
    args.root_path = root_path
    args.config_path = config_path

    network_config_path = os.path.join(config_path, args.method_name, args.config_name, 'config.yaml')
    wandb_project_name = load_yaml(network_config_path)['logging']['category']

    if args.run_id is None:
        args.run_id = generate_run_id(args)
    if args.output_dir is not None:
        args.output_dir = os.path.join(args.output_dir, 'sweep', args.run_id)
        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir, exist_ok=True)

    if args.sweep_id is None:
        setup_sweep(args, unknown_args, wandb_project_name)

    if args.agents_run_limit != 0:
        wandb.agent(args.sweep_id, project=wandb_project_name, count=args.agents_run_limit)
