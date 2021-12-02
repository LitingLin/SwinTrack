import torch
from collections.abc import Sequence, Mapping, MutableSequence, MutableMapping


def _is_sequence(object_):
    return isinstance(object_, Sequence) and not isinstance(object_, (str, bytes, bytearray, memoryview))


def get_tensors_from_object(data):
    tensor_list = []
    if _is_sequence(data):
        for i in data:
            tensor_list.extend(get_tensors_from_object(i))
    elif isinstance(data, Mapping):
        for v in data.values():
            tensor_list.extend(get_tensors_from_object(v))
    elif isinstance(data, torch.Tensor):
        tensor_list.append(data)
    return tensor_list


def replace_tensors_from_list(data, device_tensors: list):
    is_sequence = _is_sequence(data)
    if is_sequence and not isinstance(data, MutableSequence):
        data = list(data)
    if is_sequence:
        for i in range(len(data)):
            if isinstance(data[i], torch.Tensor):
                data[i] = device_tensors.pop(0)
            elif _is_sequence(data[i]) or isinstance(data[i], Mapping):
                data[i] = replace_tensors_from_list(data[i], device_tensors)
    elif isinstance(data, Mapping):
        if not isinstance(data, MutableMapping):
            data = dict(data)
        for k in data.keys():
            if isinstance(data[k], torch.Tensor):
                data[k] = device_tensors.pop(0)
            elif _is_sequence(data[k]) or isinstance(data[k], Mapping):
                data[k] = replace_tensors_from_list(data[k], device_tensors)
    elif isinstance(data, torch.Tensor):
        return device_tensors.pop(0)

    return data


def move_tensor_to_device(data, device):
    tensor_list = get_tensors_from_object(data)
    tensor_list = [tensor.to(device) for tensor in tensor_list]
    return replace_tensors_from_list(data, tensor_list)
