from core.workaround.reproducibility import seed_all_rng
from miscellanies.torch.print_running_environment import print_running_environment
from miscellanies.git_status import get_git_status_message
from miscellanies.torch.distributed import get_rank
import numpy as np
from core.run.builder import build


def build_and_run(runtime_vars, network_config, wandb_instance):
    global_synchronized_rng = np.random.Generator(np.random.PCG64(runtime_vars.conf_seed))
    runtime_vars.seed = runtime_vars.seed + get_rank()
    local_rng = np.random.Generator(np.random.PCG64(runtime_vars.seed))

    print(f"git: {get_git_status_message()}")
    print_running_environment(runtime_vars)

    print(runtime_vars)

    seed_all_rng(runtime_vars.seed)

    run = build(runtime_vars, network_config, global_synchronized_rng, local_rng, wandb_instance)
    run.run()
