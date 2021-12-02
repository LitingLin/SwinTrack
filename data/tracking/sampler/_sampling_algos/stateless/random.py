import numpy as np


# may return None when allow_insufficiency=False
def sampling_multiple_indices_with_range_and_mask(length, mask: np.ndarray=None, number_of_objects=1, sampling_range_size=None, allow_duplication=True, allow_insufficiency=True, sort=False, rng_engine: np.random.Generator=np.random.default_rng()):
    assert number_of_objects > 0
    if mask is not None:
        assert mask.dtype == np.bool_
        assert length == len(mask)

    if sampling_range_size is not None and sampling_range_size < length:
        if mask is not None:
            sampling_indices = np.arange(0, length)
            sampling_indices = sampling_indices[mask]
        else:
            sampling_indices = length
        range_mid_index = rng_engine.choice(sampling_indices)
        range_index = [range_mid_index - sampling_range_size // 2, range_mid_index + (sampling_range_size - sampling_range_size // 2)]
        range_index[0] = max(range_index[0], 0)
        range_index[1] = min(range_index[1], length)
        sampling_indices = np.arange(*range_index)
        if mask is not None:
            sampling_indices = sampling_indices[mask[range_index[0]: range_index[1]]]
    else:
        sampling_indices = np.arange(0, length)
        if mask is not None:
            sampling_indices = sampling_indices[mask]
    length_of_sampling_indices = sampling_indices.shape[0]
    if not allow_duplication:
        if length_of_sampling_indices < number_of_objects:
            if not allow_insufficiency:
                return None
            number_of_objects = length_of_sampling_indices

    indices = rng_engine.choice(sampling_indices, number_of_objects, replace=allow_duplication)
    if sort:
        indices = np.sort(indices)
    return indices


def sampling(length, rng_engine: np.random.Generator):
    return rng_engine.integers(0, length)


def sampling_with_probability(probability_array, rng_engine: np.random.Generator):
    return rng_engine.choice(np.arange(0, len(probability_array)), p=probability_array)


def sampling_with_mask(mask, rng_engine: np.random.Generator):
    assert mask.dtype == np.bool_
    indices = np.arange(0, len(mask))
    indices = indices[mask]
    return rng_engine.choice(indices)
