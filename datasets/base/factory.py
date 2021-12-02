from datasets.base.cache_service.path import prepare_dataset_cache_path
import os
from datasets.types.incompatible_error import IncompatibleError
from datasets.types.data_split import DataSplit
import copy

__all__ = ['DatasetFactory']


class _DatasetFactory:
    def __init__(self, seed, base_dataset_type, base_dataset_constructor_enum, base_filter_func, specialized_dataset_enum, specialized_dataset_type):
        self.seed = seed
        self.base_dataset_type = base_dataset_type
        self.base_dataset_constructor_enum = base_dataset_constructor_enum
        self.base_filter_func = base_filter_func
        self.specialized_dataset_enum = specialized_dataset_enum
        self.specialized_dataset_type = specialized_dataset_type

    def get_dataset_name(self):
        return f'{self.seed.name}-{self.seed.data_split.name}'

    def _try_load_from_cache(self, dataset_class, cache_extension, filters):
        cache_folder_path, cache_file_name = prepare_dataset_cache_path(dataset_class.__name__, self.seed.name, filters, self.seed.data_split)
        cache_file_path = os.path.join(cache_folder_path, cache_file_name + cache_extension)
        if os.path.exists(cache_file_path):
            try:
                dataset = dataset_class.load(cache_file_path, self.seed.root_path)
                if dataset.get_version() == self.seed.version:
                    return dataset, os.path.join(cache_folder_path, cache_file_name)
                del dataset
            except IncompatibleError:
                pass
            os.remove(cache_file_path)
        return None, os.path.join(cache_folder_path, cache_file_name)

    def construct(self, filters=None, cache_meta_data=False, dump_human_readable=False):
        if filters is not None and len(filters) == 0:
            filters = None

        dataset, cache_file_prefix = self._try_load_from_cache(self.specialized_dataset_type, '.np', filters)
        if dataset is not None:
            return dataset
        base_dataset = self.construct_as_base_interface(filters, cache_meta_data, dump_human_readable)
        dataset = base_dataset.specialize(self.specialized_dataset_enum, cache_file_prefix + '.np')
        return dataset

    @staticmethod
    def _dump_base_dataset(dataset, path):
        temp_file_path = path + '.tmp'
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)
        dataset.dump(temp_file_path)
        os.rename(temp_file_path, path)

    @staticmethod
    def _dump_base_dataset_yaml(dataset, path):
        if not os.path.exists(path):
            temp_file_path = path + '.tmp'
            if os.path.exists(temp_file_path):
                os.remove(temp_file_path)
            dataset.dump_yaml(temp_file_path)
            os.rename(temp_file_path, path)

    @staticmethod
    def _dump_cache(dataset, cache_path_prefix, dump, dump_human_readable):
        if dump:
            _DatasetFactory._dump_base_dataset(dataset, cache_path_prefix + '.p')
        if dump_human_readable:
            _DatasetFactory._dump_base_dataset_yaml(dataset, cache_path_prefix + '.yml')
        return dataset

    def _construct_base_raw(self, make_cache=False, dump_human_readable=False):
        dataset, cache_file_prefix = self._try_load_from_cache(self.base_dataset_type, '.p', None)
        if dataset is not None:
            return _DatasetFactory._dump_cache(dataset, cache_file_prefix, False, dump_human_readable)

        dataset = self.base_dataset_type(self.seed.root_path)
        with dataset.get_constructor(self.base_dataset_constructor_enum, self.seed.version) as constructor:
            constructor.set_name(self.seed.name)
            constructor.set_split(self.seed.data_split)
            self.seed.construct(constructor)
        return _DatasetFactory._dump_cache(dataset, cache_file_prefix, make_cache, dump_human_readable)

    def construct_as_base_interface(self, filters=None, make_cache=False, dump_human_readable=False):
        if filters is not None and len(filters) == 0:
            filters = None

        dataset, cache_file_prefix = self._try_load_from_cache(self.base_dataset_type, '.p', filters)
        if dataset is not None:
            return dataset
        dataset = self._construct_base_raw(make_cache, dump_human_readable)
        if filters is None:
            return dataset
        self.base_filter_func(dataset.dataset, filters)
        return _DatasetFactory._dump_cache(dataset, cache_file_prefix, make_cache, dump_human_readable)


class DatasetFactory:
    def __init__(self, seeds, base_dataset_type, base_dataset_constructor_enum, base_filter_func, specialized_dataset_enum, specialized_dataset_type):
        expanded_seeds = []
        for seed in seeds:
            if seed.data_split == DataSplit.Full:
                expanded_seeds.append(seed)
            else:
                def _check_if_data_split_in_range_and_append_to_list(data_split):
                    if seed.data_split & data_split:
                        new_seed = copy.copy(seed)
                        new_seed.data_split = data_split
                        expanded_seeds.append(new_seed)
                _check_if_data_split_in_range_and_append_to_list(DataSplit.Training)
                _check_if_data_split_in_range_and_append_to_list(DataSplit.Validation)
                _check_if_data_split_in_range_and_append_to_list(DataSplit.Testing)
                _check_if_data_split_in_range_and_append_to_list(DataSplit.Challenge)

        self.factories = [_DatasetFactory(seed, base_dataset_type, base_dataset_constructor_enum, base_filter_func, specialized_dataset_enum, specialized_dataset_type)
                          for seed in expanded_seeds]

    def construct(self, filters=None, cache_base_format=False, dump_human_readable=False):
        return tuple(factory.construct(filters, cache_base_format, dump_human_readable) for factory in self.factories)

    def construct_as_base_interface(self, filters=None, make_cache=False, dump_human_readable=False):
        return tuple(factory.construct_as_base_interface(filters, make_cache, dump_human_readable) for factory in self.factories)
