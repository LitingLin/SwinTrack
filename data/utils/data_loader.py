import torch
import torch.utils.data
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import IterableDataset
from .tensor_movement_helper import build_tensor_device_movement_helper


class _WorkerInitialization:
    def __init__(self, custom_init_fn):
        self.custom_init_fn = custom_init_fn

    def __call__(self, worker_id):
        from torch.utils.data import get_worker_info
        import numpy as np
        import random

        worker_info = get_worker_info()
        if worker_info is not None:
            worker_seed = worker_info.seed % 2 ** 32
            np.random.seed(worker_seed)
            random.seed(worker_seed)
            if self.custom_init_fn is not None:
                self.custom_init_fn(worker_info.dataset, worker_seed, worker_id)


class SetEpochHookWrapper:
    def __init__(self, object_):
        self.object = object_

    def on_epoch_begin(self, epoch):
        self.object.set_epoch(epoch)


def build_dataloader(dataset, batch_size, num_workers, device, device_index, distributed, event_register,
                     do_shuffle=True, device_tensor_selection_filter=None, worker_init_fn=None, collate_fn=None,
                     persistent_workers=False, pin_memory=False):
    pin_memory = pin_memory and 'cuda' == device

    worker_initialization_object = _WorkerInitialization(worker_init_fn)

    if isinstance(dataset, IterableDataset):
        data_loader = DataLoader(dataset, batch_size=batch_size, drop_last=False,
                                 worker_init_fn=worker_initialization_object,
                                 num_workers=num_workers, collate_fn=collate_fn, pin_memory=pin_memory,
                                 persistent_workers=persistent_workers)
    else:
        if distributed:
            sampler = DistributedSampler(dataset, shuffle=do_shuffle)
            if do_shuffle:
                event_register.register_epoch_begin_hook(SetEpochHookWrapper(sampler))
        else:
            if do_shuffle:
                sampler = torch.utils.data.RandomSampler(dataset)
            else:
                sampler = torch.utils.data.SequentialSampler(dataset)

        data_loader = DataLoader(dataset, sampler=sampler, batch_size=batch_size, drop_last=True,
                                 worker_init_fn=worker_initialization_object,
                                 num_workers=num_workers, collate_fn=collate_fn, pin_memory=pin_memory,
                                 persistent_workers=persistent_workers)

    return build_tensor_device_movement_helper(data_loader, torch.device(device), device_tensor_selection_filter, pin_memory)
