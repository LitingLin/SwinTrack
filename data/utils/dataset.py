from datasets.easy_builder.builder import build_datasets_from_config
from miscellanies.torch.distributed import is_dist_available_and_initialized, is_main_process
import torch.distributed


def build_dataset_from_config_distributed_awareness(config: dict, user_defined_parameters_handler):
    if not is_dist_available_and_initialized():
        return build_datasets_from_config(config, user_defined_parameters_handler)

    if is_main_process():
        datasets = build_datasets_from_config(config, user_defined_parameters_handler)

    torch.distributed.barrier()

    if not is_main_process():
        datasets = build_datasets_from_config(config, user_defined_parameters_handler)

    return datasets
