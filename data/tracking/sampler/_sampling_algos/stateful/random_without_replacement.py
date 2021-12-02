import numpy as np


class SamplingAlgo_RandomSamplingWithoutReplacement:
    def __init__(self, length, seed: int):
        self.position = -1
        self.rng_seed = seed
        self._shuffle(length)

    def __setstate__(self, state):
        position, length, seed = state
        self.position = position
        self.rng_seed = seed
        self._shuffle(length)

    def __getstate__(self):
        return self.position, len(self.indices), self.rng_seed

    def _shuffle(self, length):
        rng_engine = np.random.Generator(np.random.PCG64(self.rng_seed))
        self.indices = np.arange(length)
        rng_engine.shuffle(self.indices)

    def move_next(self):
        self.position += 1
        return self.position < len(self.indices)

    def current(self):
        if not 0 <= self.position < len(self.indices):
            raise IndexError
        return self.position

    def reset(self):
        self.position = -1
        self.rng_seed += 1
        self._shuffle(len(self.indices))

    def length(self):
        return len(self.indices)
