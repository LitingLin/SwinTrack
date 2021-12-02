import torch
import gc
from miscellanies.torch.tensor_movement_helper import get_tensors_from_object, replace_tensors_from_list


class DefaultTensorFilter:
    @staticmethod
    def get_tensor_list(data):
        return get_tensors_from_object(data)

    @staticmethod
    def regroup(data, device_tensors):
        return replace_tensors_from_list(data, device_tensors)


class TensorFilteringByIndices:
    def __init__(self, indices):
        self.indices = indices

    def get_tensor_list(self, data):
        split_points = []
        device_tensor_list = []
        for index in self.indices:
            datum = data[index]
            if datum is not None:
                device_tensors = get_tensors_from_object(datum)
                split_points.append(len(device_tensors))
                device_tensor_list.extend(device_tensors)
        return device_tensor_list

    def regroup(self, data, device_tensors: list):
        collated = []
        for index, datum in enumerate(data):
            if index in self.indices and datum is not None:
                datum = replace_tensors_from_list(datum, device_tensors)
            collated.append(datum)
        return collated


class CUDAPrefetchTensorMover:
    def __init__(self, iterator, device, tensor_filter=None):
        if tensor_filter is None:
            tensor_filter = DefaultTensorFilter
        self.iterator = iterator
        self.device = device
        self.tensor_filter = tensor_filter
        self.tensor_list = None

    def __len__(self):
        return len(self.iterator)

    def __iter__(self):
        self.stream = torch.cuda.Stream()
        self.iter = iter(self.iterator)
        self.preload()
        assert self.tensor_list is not None, "empty iterator"
        return self

    def __next__(self):
        torch.cuda.current_stream().wait_stream(self.stream)

        data = self.data
        tensor_list = self.tensor_list

        if data is None:
            if hasattr(self, 'stream'):
                del self.stream
            gc.collect()
            raise StopIteration

        for tensor in tensor_list:
            tensor.record_stream(torch.cuda.current_stream())
        self.preload()
        data = self.tensor_filter.regroup(data, tensor_list)
        assert len(tensor_list) == 0
        return data

    def preload(self):
        try:
            self.data = next(self.iter)
        except StopIteration:
            self.data = None
            self.tensor_list = None
            return

        self.tensor_list = self.tensor_filter.get_tensor_list(self.data)

        with torch.cuda.stream(self.stream):
            for i in range(len(self.tensor_list)):
                self.tensor_list[i] = self.tensor_list[i].to(self.device, non_blocking=True)


class TensorMover:
    def __init__(self, iterator, device, tensor_filter=None):
        if tensor_filter is None:
            tensor_filter = DefaultTensorFilter
        self.iterator = iterator
        self.device = device
        self.tensor_filter = tensor_filter

    def __len__(self):
        return len(self.iterator)

    def __iter__(self):
        self.iter = iter(self.iterator)
        return self

    def __next__(self):
        data = next(self.iter)
        tensor_list = self.tensor_filter.get_tensor_list(data)

        for i in range(len(tensor_list)):
            tensor_list[i] = tensor_list[i].to(self.device)

        data = self.tensor_filter.regroup(data, tensor_list)
        assert len(tensor_list) == 0
        return data


def build_tensor_device_movement_helper(iterator, device, tensor_filter=None, prefetch=False):
    if 'cuda' == device.type and prefetch:
        return CUDAPrefetchTensorMover(iterator, device, tensor_filter)
    else:
        return TensorMover(iterator, device, tensor_filter)
