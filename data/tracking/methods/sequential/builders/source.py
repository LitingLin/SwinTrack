from data.tracking.dataset.dataset import ForwardIteratorWrapperDataset, ForwardIteratorWrapperIterableDataset
from data.tracking.sampler.sequential.dataset_sampler import SequentialDatasetSampler, SequentialDataset
from core.run.event_dispatcher.register import EventRegister
from data.tracking.methods._common.builders.build_datasets import build_datasets
from data.tracking.methods._common.builders.build_dataset_sampler import build_dataset_sampler
from data.tracking.methods._common.builders.samplers_per_epoch import get_samples_per_epoch
from data.tracking.methods._common.builders.reproducibility import get_reproducibility_parameters
from data.tracking.methods._common.builders.build_dataloader import build_dataloader
from data.tracking.methods._common.builders.sequence_progress_tracker import SequenceProcessTracking
from .components.data_processor import build_sequential_sampling_data_processor


def build_sequential_sampling_data_source(data_config, runtime_vars, config: dict, global_synchronized_rng, local_rng, event_register: EventRegister, context):
    dataset_config = data_config['source']
    sampling_config = data_config['sampler']

    datasets, dataset_parameters = build_datasets(dataset_config)

    master_address = runtime_vars.master_address
    seed = local_rng.integers(100, 1000000)

    sequence_picker, is_random_accessible_dataset = build_dataset_sampler(datasets, dataset_parameters, sampling_config,
                                                                          master_address, global_synchronized_rng, seed,
                                                                          event_register,
                                                                          runtime_vars.rank)
    assert not is_random_accessible_dataset

    batch_size, post_processor, data_collator = build_sequential_sampling_data_processor(runtime_vars, data_config, config, datasets, context)

    samples_per_epoch = get_samples_per_epoch(datasets, sampling_config, per_frame=True, align=batch_size, drop_last=False)

    rng_engine_seed, reset_per_epoch = get_reproducibility_parameters(sampling_config)

    samplers = [SequentialDatasetSampler(sequence_picker, datasets) for _ in range(batch_size)]
    dataset = SequentialDataset(samplers, post_processor)

    if sampling_config['dataset_sampling']['type'] == 'run_through':
        torch_dataset = ForwardIteratorWrapperIterableDataset(dataset, samples_per_epoch, rng_engine_seed)
        worker_init_fn = ForwardIteratorWrapperIterableDataset.worker_init_function
    else:
        torch_dataset = ForwardIteratorWrapperDataset(dataset, samples_per_epoch, rng_engine_seed)
        worker_init_fn = ForwardIteratorWrapperDataset.worker_init_function

    if reset_per_epoch:
        event_register.register_epoch_begin_hook(torch_dataset)

    dataloader = build_dataloader(batch_size, runtime_vars, torch_dataset, event_register,
                                  worker_init_fn, data_collator)

    if sampling_config['dataset_sampling']['type'] == 'run_through' and runtime_vars.num_workers > 1:
        from ..pipeline.common import IterableDatasetWorkerDataFilter
        dataloader = IterableDatasetWorkerDataFilter(dataloader, runtime_vars.num_workers)

    data_pipeline = None

    if sampling_config['dataset_sampling']['type'] == 'run_through':
        dataloader = SequenceProcessTracking(dataloader, sequence_picker)
        data_pipeline = {'data_pipeline': (dataloader,)}

    context['iterations_per_epoch'] = len(dataloader)

    return dataloader, data_pipeline
