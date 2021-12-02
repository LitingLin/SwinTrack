from data.tracking.sampler.SiamFC.dataset_sampler import SOTTrackingSiameseForwardIteratorDatasetSampler,SOTTrackingSiameseRandomAccessIteratorDatasetSampler
from data.tracking.methods.SiamFC.common.image_decoding import SiamFCImageDecodingProcessor
from data.tracking.sampler.SiamFC.type import SiamesePairSamplingMethod
from data.tracking.dataset.dataset import ForwardIteratorWrapperDataset, RandomAccessIteratorWrapperDataset
from core.run.event_dispatcher.register import EventRegister
from data.tracking.methods._common.builders.build_datasets import build_datasets
from data.tracking.methods._common.builders.build_dataset_sampler import build_dataset_sampler
from data.tracking.methods._common.builders.samplers_per_epoch import get_samples_per_epoch
from data.tracking.methods._common.builders.dataset_sampling_weight import get_dataset_sampling_weight
from data.tracking.methods._common.builders.reproducibility import get_reproducibility_parameters


def build_siamfc_dataset(sampling_config: dict, dataset_config: dict, post_processor,
                         master_address, deterministic_rng,
                         seed, hook_register: EventRegister, rank):
    datasets, dataset_parameters = build_datasets(dataset_config)

    sequence_picker, is_random_accessible_dataset = build_dataset_sampler(datasets, dataset_parameters, sampling_config,
                                                                          master_address, deterministic_rng, seed,
                                                                          hook_register, rank)

    samples_per_epoch = get_samples_per_epoch(datasets, sampling_config)
    dataset_sampling_weights = get_dataset_sampling_weight(dataset_parameters)

    rng_engine_seed, reset_per_epoch = get_reproducibility_parameters(sampling_config)

    negative_sample_ratio = 0
    sequence_sampler_parameters = sampling_config['sequence_sampling']['parameters']
    negative_sample_random_picking_ratio = 0.
    if 'negative_sample_random_picking_ratio' in sequence_sampler_parameters:
        negative_sample_random_picking_ratio = sequence_sampler_parameters['negative_sample_random_picking_ratio']
    default_frame_range = sequence_sampler_parameters['frame_range']
    sequence_sampling_method = SiamesePairSamplingMethod[sequence_sampler_parameters['sampling_method']]
    enforce_fine_positive_sample = False
    if 'enforce_fine_positive_sample' in sequence_sampler_parameters:
        enforce_fine_positive_sample = sequence_sampler_parameters['enforce_fine_positive_sample']
    enable_adaptive_frame_range = True
    if 'enable_adaptive_frame_range' in sequence_sampler_parameters:
        enable_adaptive_frame_range = sequence_sampler_parameters['enable_adaptive_frame_range']

    if 'negative_sample_ratio' in sampling_config:
        negative_sample_ratio = sampling_config['negative_sample_ratio']

    dataset_sampling_parameters = [{}] * len(datasets)
    for dataset_parameter, dataset_sampling_parameter in zip(dataset_parameters, dataset_sampling_parameters):
        if 'sampling' in dataset_parameter:
            sampling_parameters = dataset_parameter['sampling']
            if 'frame_range' in sampling_parameters:
                dataset_sampling_parameter['frame_range'] = sampling_parameters['frame_range']

    processor = SiamFCImageDecodingProcessor(post_processor)

    if is_random_accessible_dataset:
        assert not sequence_sampler_parameters['enforce_fine_positive_sample']

    if is_random_accessible_dataset:
        sampler = SOTTrackingSiameseRandomAccessIteratorDatasetSampler(
            sequence_picker, datasets, negative_sample_ratio, enforce_fine_positive_sample,
            sequence_sampling_method, enable_adaptive_frame_range,
            negative_sample_random_picking_ratio,
            default_frame_range,
            dataset_sampling_parameters, dataset_sampling_weights, processor)
        dataset = RandomAccessIteratorWrapperDataset(sampler, samples_per_epoch, rng_engine_seed)
    else:
        sampler = SOTTrackingSiameseForwardIteratorDatasetSampler(
            sequence_picker, datasets, negative_sample_ratio, enforce_fine_positive_sample,
            sequence_sampling_method, enable_adaptive_frame_range,
            negative_sample_random_picking_ratio,
            default_frame_range,
            dataset_sampling_parameters, dataset_sampling_weights, processor)
        dataset = ForwardIteratorWrapperDataset(sampler, samples_per_epoch, rng_engine_seed)

    if reset_per_epoch:
        hook_register.register_epoch_begin_hook(dataset)

    return dataset, dataset.worker_init_function
