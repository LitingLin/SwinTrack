import copy
import os
import wandb

from miscellanies.flatten_dict.flattern_dict import flatten
from miscellanies.git_status import get_git_status
from miscellanies.torch.distributed import is_dist_available_and_initialized


def setup_wandb(args, network_config: dict, notes):
    tags = network_config['logging']['tags']
    mode = 'online' if not args.wandb_run_offline else 'offline'

    output_dir = args.output_dir
    if output_dir is None:
        output_dir = os.path.join(args.root_path, 'logging')
    os.makedirs(output_dir, exist_ok=True)

    group = None
    config = None
    project = None
    run_id = None

    if not args.do_sweep:
        project = network_config['logging']['category']
        run_id = args.run_id

        network_config = copy.deepcopy(network_config)
        assert 'runtime_vars' not in network_config
        network_config['runtime_vars'] = vars(args)

        config = flatten(network_config, reducer='dot', enumerate_types=(list,))
        config['git_version'] = get_git_status()

        if args.wandb_distributed_aware and is_dist_available_and_initialized():
            group = run_id
            run_id = run_id + f'-rank{args.rank // args.local_world_size}.{args.local_rank}'

    if run_id is not None:
        if len(run_id) > 128:
            run_id = run_id[:128]
            print('warning: run id truncated for wandb limitation')

    wandb_instance = wandb.init(project=project, tags=tags, config=config, force=True, job_type='train', id=run_id, mode=mode, dir=output_dir, group=group, notes=notes)
    return wandb_instance
