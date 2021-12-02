import torch


class FeatureMapCache:
    def __init__(self, max_cache_length, dim, template_size):
        self.shape = (max_cache_length, dim, template_size[1], template_size[0])
        self.device = torch.device('cpu')
        self.cache = None

    def create(self):
        self.cache = torch.empty(self.shape, dtype=torch.float, device=self.device)

    def close(self):
        del self.cache
        self.cache = None

    def to(self, device):
        self.device = device
        if self.cache is not None:
            self.cache = self.cache.to(device)

    def put(self, index, tensor):
        self.cache[index, ...] = tensor

    def put_batch(self, indices, tensor_list):
        if isinstance(tensor_list, torch.Tensor):
            self.cache[indices, ...] = tensor_list
        else:
            for index, tensor in indices, tensor_list:
                self.cache[index, ...] = tensor

    def get_all(self):
        return self.cache

    def get_batch(self, indices):
        return self.cache[indices, ...]

    def get(self, index):
        return self.cache[index, ...]


class TokenCache:
    def __init__(self, max_cache_length, dim, num_tokens):
        self.shape = (max_cache_length, num_tokens, dim)
        self.device = torch.device('cpu')
        self.cache = None

    def create(self):
        self.cache = torch.empty(self.shape, dtype=torch.float, device=self.device)

    def close(self):
        del self.cache
        self.cache = None

    def to(self, device):
        self.device = device
        if self.cache is not None:
            self.cache = self.cache.to(device)

    def put(self, index, tensor):
        self.cache[index, ...] = tensor

    def put_batch(self, indices, tensor_list):
        if isinstance(tensor_list, torch.Tensor):
            self.cache[indices, ...] = tensor_list
        else:
            for index, tensor in indices, tensor_list:
                self.cache[index, ...] = tensor

    def get_all(self):
        return self.cache

    def get_batch(self, indices):
        return self.cache[indices, ...]

    def get(self, index):
        return self.cache[index, ...]


class ScalerCache:
    def __init__(self, max_cache_length, dim):
        self.shape = (max_cache_length, dim)
        self.device = torch.device('cpu')
        self.cache = None

    def create(self):
        self.cache = torch.empty(self.shape, dtype=torch.float, device=self.device)

    def close(self):
        del self.cache
        self.cache = None

    def to(self, device):
        self.device = device
        if self.cache is not None:
            self.cache = self.cache.to(device)

    def put(self, index, tensor):
        self.cache[index, ...] = tensor

    def put_batch(self, indices, tensor_list):
        if isinstance(tensor_list, torch.Tensor):
            self.cache[indices, ...] = tensor_list
        else:
            for index, tensor in indices, tensor_list:
                self.cache[index, ...] = tensor

    def get_all(self):
        return self.cache

    def get_batch(self, indices):
        return self.cache[indices, ...]

    def get(self, index):
        return self.cache[index, ...]


class MultiScaleTokenCache:
    def __init__(self, cache_length, dim_list, num_tokens_list):
        self.shape_list = tuple((cache_length, num_tokens, dim) for num_tokens, dim in zip(num_tokens_list, dim_list))
        self.device = torch.device('cpu')
        self.cache_list = None

    def create(self):
        self.cache_list = tuple(torch.empty(shape, dtype=torch.float, device=self.device) for shape in self.shape_list)

    def close(self):
        del self.cache_list
        self.cache_list = None

    def to(self, device):
        self.device = device
        if self.cache_list is not None:
            for i in range(len(self.cache_list)):
                self.cache_list[i] = self.cache_list[i].to(device)

    def put(self, index, tensor_list):
        assert len(tensor_list) == len(self.cache_list)
        for cache, tensor in zip(self.cache_list, tensor_list):
            cache[index, ...] = tensor

    def put_batch(self, indices, tensor_list):
        assert len(tensor_list) == len(self.cache_list)
        for cache, tensor in zip(self.cache_list, tensor_list):
            if isinstance(tensor, torch.Tensor):
                cache[indices, ...] = tensor
            else:
                for index, sub_tensor in zip(indices, tensor):
                    cache[index, ...] = sub_tensor

    def get_all(self):
        return self.cache_list

    def get_batch(self, indices):
        return tuple(cache[indices, ...] for cache in self.cache_list)

    def get(self, index):
        return tuple(cache[index, ...] for cache in self.cache_list)


class UUIDBasedCacheService:
    def __init__(self, max_cache_length, cache):
        self.uuid_list = [None] * max_cache_length
        self.free_bits = [True] * max_cache_length
        self.cache = cache

    def create(self):
        self.cache.create()

    def close(self):
        self.cache.close()

    def to(self, device):
        self.cache.to(device)

    def put(self, uuid: str, item):
        index = self.uuid_list.index(uuid) if uuid in self.uuid_list else None
        if index is not None:
            assert self.free_bits[index]
        else:
            index = self.free_bits.index(True)
            self.uuid_list[index] = uuid
            self.free_bits[index] = False
        self.cache.put(index, item)

    def put_batch(self, uuids, items):
        indices = []
        for uuid in uuids:
            index = self.uuid_list.index(uuid) if uuid in self.uuid_list else None
            if index is not None:
                assert self.free_bits[index]
            else:
                index = self.free_bits.index(True)
                self.uuid_list[index] = uuid
                self.free_bits[index] = False
            indices.append(index)
        self.cache.put_batch(indices, items)

    def get_all(self):
        return self.cache.get_all()

    def release(self, uuid):
        index = self.uuid_list.index(uuid)
        self.free_bits[index] = True

    def get(self, uuid):
        index = self.uuid_list.index(uuid)
        return self.cache.get(index)

    def get_batch(self, uuids):
        indices = [self.uuid_list.index(uuid) for uuid in uuids]
        return self.cache.get_batch(indices)
