import data.utils.data_loader
from data.utils.tensor_movement_helper import TensorFilteringByIndices


def build_dataloader(batch_size, runtime_vars, dataset, event_register, worker_init_fn, collate_fn):
    do_shuffle = False  # randomness has been provided in other places
    device_tensor_filter = TensorFilteringByIndices((0, 1, 3))

    return data.utils.data_loader.build_dataloader(dataset, batch_size, runtime_vars.num_workers, runtime_vars.device,
                                                   runtime_vars.local_rank, runtime_vars.distributed,
                                                   event_register, do_shuffle, device_tensor_filter,
                                                   worker_init_fn, collate_fn, runtime_vars.persistent_data_workers,
                                                   runtime_vars.pin_memory)
