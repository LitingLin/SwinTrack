import yaml
import os
import copy
import importlib
from datasets.types.data_split import DataSplit
from data.types.bounding_box_format import BoundingBoxFormat


def merge_config(default: dict, user_defined: dict):
    merged = {}
    for key, user_defined_value in user_defined.items():
        if user_defined_value is None:
            user_defined_value = {}
        if key in default:
            default_value = default[key]
            default_value = copy.copy(default_value)
            default_value.update(user_defined_value)
            user_defined_value = default_value
        merged[key] = user_defined_value

    return merged


def parseDataSplit(split_string: str):
    if split_string == 'train':
        return DataSplit.Training
    elif split_string == 'val':
        return DataSplit.Validation
    elif split_string == 'test':
        return DataSplit.Testing
    elif split_string == 'full':
        return DataSplit.Full
    else:
        raise Exception('Invalid value {}'.format(split_string))


def get_data_split_from_config(split_strings: list):
    split = parseDataSplit(split_strings[0])
    if len(split_strings) > 1:
        for split_string in split_strings[1:]:
            split |= parseDataSplit(split_string)
    return split


known_parameters = ('type', 'splits', 'parameters', 'filters')


def get_unknown_parameters(dataset_building_parameters: dict):
    return {key: value for key, value in dataset_building_parameters.items() if key not in known_parameters}


def parse_filters(config):
    filters = []
    for filter_key, filter_value in config.items():
        if filter_key == 'DataCleaning':
            for filter_data_cleaning_key, filter_data_cleaning_value in filter_value.items():
                module = importlib.import_module('datasets.filter.DataCleaning.{}'.format(filter_data_cleaning_key))
                filter_class = getattr(module, 'DataCleaning_{}'.format(filter_data_cleaning_key))
                filters.append(filter_class(**filter_data_cleaning_value))
        else:
            module = importlib.import_module('datasets.filter.{}'.format(filter_key))
            filter_class = getattr(module, filter_key)
            filters.append(filter_class(**filter_value))
    return filters


def _default_unknown_parameter_handler(datasets, parameters):
    import copy
    return tuple(copy.deepcopy(parameters) for _ in range(len(datasets)))


def build_datasets(config: dict, unknown_parameter_handler=_default_unknown_parameter_handler):
    filters = []
    if 'filters' in config:
        dataset_filter_names = config['filters']
        filters.extend(parse_filters(dataset_filter_names))

    datasets = []

    constructor_params = {}
    if 'config' in config:
        dataset_building_config = config['config']
        if 'bounding_box' in dataset_building_config:
            bounding_box_building_config = dataset_building_config['bounding_box']
            if 'format' in bounding_box_building_config:
                constructor_params['bounding_box_format'] = BoundingBoxFormat[bounding_box_building_config['format']]
        if 'dump_human_readable' in config:
            constructor_params['dump_human_readable'] = config['dump_human_readable']
        if 'cache_meta_data' in config:
            constructor_params['cache_meta_data'] = config['cache_meta_data']

    extra_parameters = []

    for dataset_name, dataset_building_parameter in config['datasets'].items():
        dataset_type = dataset_building_parameter['type']
        path = None
        if 'path' in dataset_building_parameter:
            path = dataset_building_parameter['path']
        if dataset_type == 'SOT':
            from datasets.SOT.factory import SingleObjectTrackingDatasetFactory
            module = importlib.import_module('datasets.SOT.seed.{}'.format(dataset_name))
            factory_class = SingleObjectTrackingDatasetFactory
        elif dataset_type == 'MOT':
            from datasets.MOT.factory import MultipleObjectTrackingDatasetFactory
            module = importlib.import_module('datasets.MOT.seed.{}'.format(dataset_name))
            factory_class = MultipleObjectTrackingDatasetFactory
        elif dataset_type == 'DET':
            from datasets.DET.factory import DetectionDatasetFactory
            module = importlib.import_module('datasets.DET.seed.{}'.format(dataset_name))
            factory_class = DetectionDatasetFactory
        else:
            raise Exception('Unsupported dataset type {}'.format(dataset_type))

        seed_class = getattr(module, '{}_Seed'.format(dataset_name))

        if 'parameters' in dataset_building_parameter:
            seed_parameters = dataset_building_parameter['parameters']
            seed = seed_class(root_path=path, **seed_parameters)
        else:
            seed = seed_class(root_path=path)

        seed.data_split = get_data_split_from_config(dataset_building_parameter['splits'])
        factory = factory_class([seed])

        if 'filters' in dataset_building_parameter:
            dataset_filters = parse_filters(dataset_building_parameter['filters'])
            dataset_filters.extend(filters)
        else:
            dataset_filters = filters

        if len(dataset_filters) == 0:
            dataset_filters = None

        dataset = factory.construct(dataset_filters, **constructor_params)

        extra_parameters.extend(unknown_parameter_handler(dataset, get_unknown_parameters(dataset_building_parameter)))
        datasets.extend(dataset)

    return datasets, extra_parameters


def build_datasets_from_config(config: dict, unknown_parameter_handler=_default_unknown_parameter_handler):
    with open(os.path.join(os.path.dirname(__file__), 'defaults.yaml'), 'rb') as fid:
        default = yaml.safe_load(fid)
    default = default['datasets']
    dataset_config = merge_config(default, config['datasets'])
    config['datasets'] = dataset_config
    return build_datasets(config, unknown_parameter_handler)
