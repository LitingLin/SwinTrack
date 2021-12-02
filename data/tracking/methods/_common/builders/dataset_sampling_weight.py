import numpy as np


def get_dataset_sampling_weight(dataset_parameters):
    dataset_sampling_weights = np.empty(len(dataset_parameters), dtype=np.float64)
    for index, dataset_parameter in enumerate(dataset_parameters):
        sampling_weight = 1
        if 'sampling' in dataset_parameter:
            sampling_parameters = dataset_parameter['sampling']
            if 'weight' in sampling_parameters:
                sampling_weight *= sampling_parameters['weight']
        dataset_sampling_weights[index] = sampling_weight

    dataset_sampling_weights = dataset_sampling_weights / dataset_sampling_weights.sum()
    return dataset_sampling_weights
