from data.utils.dataset import build_dataset_from_config_distributed_awareness
import copy
import numpy as np


def _customized_dataset_parameter_handler(datasets, parameters):
    number_of_datasets = len(datasets)
    datasets_parameters = [copy.deepcopy(parameters) for _ in range(number_of_datasets)]

    datasets_weight = np.array([len(dataset) for dataset in datasets], dtype=np.float64)
    datasets_weight = datasets_weight / datasets_weight.sum()

    if 'Sampling' in parameters:
        sampling_parameters = parameters['Sampling']
        if 'weight' in sampling_parameters:
            for index, dataset_parameters in enumerate(datasets_parameters):
                dataset_parameters['Sampling']['weight'] = float(sampling_parameters['weight'] * datasets_weight[index])

    return datasets_parameters


def build_datasets(dataset_config):
    datasets, dataset_parameters = build_dataset_from_config_distributed_awareness(dataset_config, _customized_dataset_parameter_handler)
    return datasets, dataset_parameters
