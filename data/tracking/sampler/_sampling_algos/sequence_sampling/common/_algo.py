import numpy as np
from data.tracking.sampler._sampling_algos.stateless.random import sampling, sampling_with_mask


def sample_one_positive(length, mask, rng_engine: np.random.Generator):
    if mask is None:
        z_index = sampling(length, rng_engine)
    else:
        z_index = sampling_with_mask(mask, rng_engine)

    return z_index
