from core.run.event_dispatcher.register import EventRegister
from data.tracking.methods._common.builders.build_dataloader import build_dataloader
from .components.dataset import build_siamfc_dataset
from .components.data_processor import build_siamfc_tracker_data_processor


def build_siamfc_data_source(data_config: dict, runtime_vars, config: dict,
                             global_synchronized_rng, local_rng,
                             event_register: EventRegister, context):
    data_processor, data_batch_collator, metric_collector = build_siamfc_tracker_data_processor(data_config, config)

    master_address = runtime_vars.master_address
    seed = local_rng.integers(100, 1000000)
    dataset_config = data_config['source']
    sampling_config = data_config['sampler']
    dataset, worker_init_fn = build_siamfc_dataset(sampling_config, dataset_config, data_processor,
                                                   master_address, global_synchronized_rng, seed, event_register,
                                                   runtime_vars.rank)

    dataloader = build_dataloader(data_config['batch_size'], runtime_vars, dataset, event_register,
                                  worker_init_fn, data_batch_collator)

    context['iterations_per_epoch'] = len(dataloader)
    context['batch_size'] = data_config['batch_size']

    return dataloader, {'data_pipeline': (metric_collector, )}
