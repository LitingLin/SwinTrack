import numpy as np
import pickle
import os
from typing import Union, Iterable


class ByteArrayMemoryMapped:
    def __init__(self, path: str):
        storage: np.ndarray = np.memmap(path, mode='r')
        assert np.array_equal(np.array(memoryview(b"strg"), dtype=np.uint8), storage[0: 4]), 'not a storage file'
        indices_size = int.from_bytes(storage[4: 8], byteorder='little', signed=False)
        indices = pickle.loads(storage[8: 8 + indices_size])
        self.storage = storage
        self.offset = indices_size + 8
        self.indices = indices

    def __getitem__(self, index: int):
        offset = self.indices[index] + self.offset
        size = self.indices[index + 1] - self.indices[index]
        return memoryview(self.storage[offset: offset + size])

    def __len__(self):
        return len(self.indices)


class ByteArrayMemoryMappedConstructor:
    def __init__(self, path: str):
        self.indices = []
        self.bytes = []
        self.path = path
        self.length = 0

    def append_array(self, bytes_: Iterable[bytes]):
        self.indices.append(self.length)

        length = 0
        for b in bytes_:
            self.bytes.append(np.frombuffer(b, dtype=np.uint8))
            length += len(b)

        self.length += length

    def append(self, bytes_: bytes):
        self.indices.append(self.length)
        self.bytes.append(np.frombuffer(bytes_, dtype=np.uint8))
        length = len(bytes_)
        self.length += length

    def construct(self):
        self.indices.append(self.length)
        indices_bytes = pickle.dumps(self.indices)
        indices_length_bytes = int.to_bytes(len(indices_bytes), length=4, byteorder='little', signed=False)
        raw_bytes = np.concatenate([np.array(memoryview(b"strg"), dtype=np.uint8),
                                    np.frombuffer(indices_length_bytes, dtype=np.uint8),
                                    np.frombuffer(indices_bytes, dtype=np.uint8)] + self.bytes)

        path = self.path + '.lock'
        if os.path.exists(path):
            os.remove(path)

        raw_bytes_on_disk = np.memmap(path, mode='w+', shape=raw_bytes.shape)
        raw_bytes_on_disk[:] = raw_bytes[:]
        del raw_bytes_on_disk

        if os.path.exists(self.path):
            os.remove(self.path)
        os.rename(path, self.path)

        del self.indices
        del self.bytes

        return ByteArrayMemoryMapped(self.path)


class ListMemoryMapped:
    def __init__(self, engine: Union[str, ByteArrayMemoryMapped]):
        if isinstance(engine, str):
            self.engine = ByteArrayMemoryMapped(engine)
        elif isinstance(engine, ByteArrayMemoryMapped):
            self.engine = engine
        else:
            raise Exception('unknown parameter type')

    def __getitem__(self, index: int):
        raw_bytes = self.engine[index]
        type_ = int.from_bytes(raw_bytes[0: 4], byteorder='little', signed=False)
        raw_bytes = raw_bytes[4:]
        if type_ == 0:
            return pickle.loads(raw_bytes)
        elif type_ == 1:
            numpy_desc_size = int.from_bytes(raw_bytes[0: 4], byteorder='little', signed=False)
            numpy_desc = pickle.loads(raw_bytes[4: 4 + numpy_desc_size])
            return np.frombuffer(raw_bytes[4 + numpy_desc_size:], dtype=numpy_desc[0]).reshape(numpy_desc[1])
        else:
            raise Exception('Unsupported')

    def __len__(self):
        return len(self.engine)


class ListMemoryMappedConstructor:
    def __init__(self, path):
        self.constructor = ByteArrayMemoryMappedConstructor(path)

    def append(self, object_):
        if isinstance(object_, np.ndarray):
            type_ = 1
            numpy_desc_bytes = pickle.dumps((object_.dtype, object_.shape))
            numpy_desc_size = len(numpy_desc_bytes)
            numpy_desc_size_bytes = numpy_desc_size.to_bytes(length=4, byteorder='little', signed=False)
            self.constructor.append_array(
                [type_.to_bytes(length=4, byteorder='little', signed=False), numpy_desc_size_bytes,
                 numpy_desc_bytes, memoryview(object_).cast('B')])
        else:
            type_ = 0
            self.constructor.append_array(
                [type_.to_bytes(length=4, byteorder='little', signed=False), pickle.dumps(object_)])

    def construct(self):
        return ListMemoryMapped(self.constructor.construct())
