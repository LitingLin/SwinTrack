import torch
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from torchvision import transforms


def get_transform():
    return transforms.Normalize(mean=torch.tensor(IMAGENET_DEFAULT_MEAN), std=torch.tensor(IMAGENET_DEFAULT_STD))


class IterableDatasetWorkerDataFilter:
    def __init__(self, data_loader, num_workers):
        assert num_workers > 1
        self.data_loader = data_loader
        self.exhausted_flag = None
        self.num_workers = num_workers

    def __iter__(self):
        self.data_loader_iter = iter(self.data_loader)
        self.exhausted_flag = torch.zeros((self.num_workers, ), dtype=torch.bool)
        self.index = 0
        return self

    def __next__(self):
        while True:
            current_index = self.index
            self.index += 1
            if self.index >= self.num_workers:
                self.index = 0
            sample, label, misc_on_host, misc_on_device = next(self.data_loader_iter)

            if torch.all(self.exhausted_flag):
                return sample, label, misc_on_host, misc_on_device
            if sample is None:
                self.exhausted_flag[current_index] = True
                continue
            return sample, label, misc_on_host, misc_on_device

    def __len__(self):
        return len(self.data_loader)
