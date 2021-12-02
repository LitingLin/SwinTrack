from ...pipeline.worker import SequentialWorkerDataProcessor, SequentialSamplingWorkerDataCollator
from datasets.SOT.dataset import SingleObjectTrackingDataset_MemoryMapped


def build_sequential_sampling_data_processor(runtime_vars, data_config, config: dict, datasets, context):
    assert all(isinstance(dataset, SingleObjectTrackingDataset_MemoryMapped) for dataset in datasets)
    dataset_sequence_names = {}

    total_num_sequences = 0
    for dataset in datasets:
        sequence_names = [sequence.get_name() for sequence in dataset]
        dataset_sequence_names[dataset.get_unique_id()] = sequence_names
        total_num_sequences += len(sequence_names)

    context['dataset_sequence_names'] = dataset_sequence_names
    context['total_num_sequences'] = total_num_sequences

    if 'batch_size' not in data_config:
        assert 'max_batch_size' in data_config
        max_batch_size = data_config['max_batch_size']
        world_size = runtime_vars.world_size
        if world_size is None:
            world_size = 1
        scheduling_capacity = runtime_vars.num_workers * world_size
        batch_size = max(1, min(max_batch_size, total_num_sequences // scheduling_capacity))
    else:
        batch_size = data_config['batch_size']

    context['batch_size'] = batch_size

    network_data_config = config['data']

    post_processor = SequentialWorkerDataProcessor(batch_size,
                                                   data_config['tracking']['pre_processing']['template']['area_factor'],
                                                   network_data_config['template_size'],
                                                   network_data_config['interpolation_mode'])

    return batch_size, post_processor, SequentialSamplingWorkerDataCollator()
