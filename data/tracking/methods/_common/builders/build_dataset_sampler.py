from core.run.event_dispatcher.register import EventRegister
from miscellanies.torch.distributed import is_main_process
from data.tracking.methods._common.builders.dataset_sampling_weight import get_dataset_sampling_weight


class _HookWrapper:
    def __init__(self, object_):
        self.object_ = object_

    def on_started(self):
        self.object_.start()

    def on_finished(self):
        self.object_.stop()

    def on_epoch_begin(self, _):
        self.object_.reset()


def build_dataset_sampler(datasets, dataset_parameters, sampling_config: dict,
                          master_address, deterministic_rng, seed, hook_register: EventRegister, rank):
    dataset_sampling_config = sampling_config['dataset_sampling']

    if dataset_sampling_config['type'] == 'random_without_replacement':
        from data.tracking.sampler._sampling_algos.sequence_picking.random_without_replacement.random_sampler import RandomSequencePickingOrchestrationServer, RandomSequencePickingClient

        if 'parameters' not in dataset_sampling_config or 'listen_address' not in dataset_sampling_config['parameters']:
            sampling_orchestrator_server_address = f'tcp://{master_address}:{deterministic_rng.integers(10000, 50000)}'
        else:
            sampling_orchestrator_server_address = dataset_sampling_config['parameters']['listen_address']
        if is_main_process():
            dataset_sampling_weights = get_dataset_sampling_weight(dataset_parameters)
            sampler_server = RandomSequencePickingOrchestrationServer(datasets, dataset_sampling_weights,
                                                                      sampling_orchestrator_server_address, seed)
            sampler_server_hook_wrapper = _HookWrapper(sampler_server)
            hook_register.register_started_hook(sampler_server_hook_wrapper)
            hook_register.register_finished_hook(sampler_server_hook_wrapper)
            hook_register.register_stateful_object('random_without_replacement_sampling_orchestrator', sampler_server)
            hook_register.register_status_collector('random_without_replacement_sampling_orchestrator', sampler_server)
        sequence_picker = RandomSequencePickingClient(sampling_orchestrator_server_address)
        random_accessible_dataset = False
    elif dataset_sampling_config['type'] == 'pre_shuffled':
        from data.tracking.sampler._sampling_algos.sequence_picking.pre_shuffle.sampler import PreShuffledSequencePicker
        from .samplers_per_epoch import get_samples_per_epoch

        samples_per_epoch = get_samples_per_epoch(datasets, sampling_config)
        dataset_sampling_weights = get_dataset_sampling_weight(dataset_parameters)
        sequence_picker = PreShuffledSequencePicker(datasets, dataset_sampling_weights, samples_per_epoch, seed)
        random_accessible_dataset = True
    elif dataset_sampling_config['type'] == 'run_through':
        from data.tracking.sampler._sampling_algos.sequence_picking.run_through.random_sampler import RunThroughSequencePickingOrchestrationServer, RunThroughSequencePickingClient

        if 'parameters' not in dataset_sampling_config or 'listen_address' not in dataset_sampling_config['parameters']:
            sampling_orchestrator_server_address = f'tcp://{master_address}:{deterministic_rng.integers(10000, 50000)}'
        else:
            sampling_orchestrator_server_address = dataset_sampling_config['parameters']['listen_address']
        if is_main_process():
            sampler_server = RunThroughSequencePickingOrchestrationServer(datasets, sampling_orchestrator_server_address, seed)
            sampler_server_hook_wrapper = _HookWrapper(sampler_server)
            hook_register.register_started_hook(sampler_server_hook_wrapper)
            hook_register.register_finished_hook(sampler_server_hook_wrapper)
            hook_register.register_epoch_begin_hook(sampler_server_hook_wrapper)
        sequence_picker = RunThroughSequencePickingClient(sampling_orchestrator_server_address, rank)
        random_accessible_dataset = False
    else:
        raise NotImplementedError(dataset_sampling_config['type'])

    return sequence_picker, random_accessible_dataset
