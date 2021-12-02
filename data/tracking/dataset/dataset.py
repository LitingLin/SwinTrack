import torch.utils.data
import numpy as np
import torch


class _BaseWrapperDataset(torch.utils.data.Dataset):
    def __init__(self, sampler, samples_per_epoch, fixed_seed=None):
        self.dataset_sampler = sampler
        self.samples_per_epoch = samples_per_epoch
        self.rng_engine = np.random.Generator(np.random.PCG64())
        self.fixed_seed = fixed_seed

    @staticmethod
    def worker_init_function(class_object, seed, worker_id):
        if class_object.fixed_seed is not None:
            class_object.seed_rng_engine(seed)

    def on_epoch_begin(self, _):
        if self.fixed_seed is not None:
            self.seed_rng_engine(self.fixed_seed)

    def seed_rng_engine(self, seed):
        self.rng_engine = np.random.Generator(np.random.PCG64(seed))

    def __len__(self):
        return self.samples_per_epoch


class ForwardIteratorWrapperDataset(_BaseWrapperDataset):
    def __init__(self, sampler, samples_per_epoch, fixed_seed=None):
        super(ForwardIteratorWrapperDataset, self).__init__(sampler, samples_per_epoch, fixed_seed)

    def __getitem__(self, _):
        return self.dataset_sampler.get_next(self.rng_engine)


class RandomAccessIteratorWrapperDataset(_BaseWrapperDataset):
    def __init__(self, sampler, samples_per_epoch, fixed_seed=None):
        super(RandomAccessIteratorWrapperDataset, self).__init__(sampler, samples_per_epoch, fixed_seed)

    def __getitem__(self, index):
        return self.dataset_sampler.get(index, self.rng_engine)


class ForwardIteratorWrapperIterableDataset(torch.utils.data.IterableDataset):
    def __init__(self, sampler, estimated_samples_per_epoch, fixed_seed=None):
        self.dataset_sampler = sampler
        self.estimated_samples_per_epoch = estimated_samples_per_epoch
        self.rng_engine = np.random.Generator(np.random.PCG64())
        self.fixed_seed = fixed_seed

    def on_epoch_begin(self, _):
        if self.fixed_seed is not None:
            self.seed_rng_engine(self.fixed_seed)

    def __len__(self):
        return self.estimated_samples_per_epoch

    def __iter__(self):
        return self

    def __next__(self):
        return self.dataset_sampler.get_next(self.rng_engine)

    def seed_rng_engine(self, seed):
        self.rng_engine = np.random.Generator(np.random.PCG64(seed))

    @staticmethod
    def worker_init_function(class_instance, seed, worker_id):
        if class_instance.fixed_seed is not None:
            class_instance.seed_rng_engine(seed)
