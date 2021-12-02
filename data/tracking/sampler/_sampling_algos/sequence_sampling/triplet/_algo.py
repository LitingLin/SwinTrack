import numpy as np
from data.tracking.sampler.SiamFC.type import SiamesePairSamplingMethod
from data.tracking.sampler._sampling_algos.stateless.random import sampling_multiple_indices_with_range_and_mask
from data.tracking.sampler._sampling_algos.sequence_sampling.common._algo import sample_one_positive


def do_triplet_sampling_positive_only(length: int, frame_range: int, aux_frame_range: int, mask: np.ndarray=None,
                                      sampling_method: SiamesePairSamplingMethod=SiamesePairSamplingMethod.causal,
                                      aux_sampling_method: SiamesePairSamplingMethod=SiamesePairSamplingMethod.causal,
                                      rng_engine: np.random.Generator=np.random.default_rng()):
    assert frame_range > 0
    assert aux_frame_range > 0
    sort = False
    frame_range = frame_range + 1
    if sampling_method == SiamesePairSamplingMethod.causal:
        sort = True
    indices = sampling_multiple_indices_with_range_and_mask(length, mask, 2, frame_range, allow_duplication=False, allow_insufficiency=True, sort=sort, rng_engine=rng_engine)
    if len(indices) < 2:
        return indices
    elif len(indices) == 2:
        x_index = indices[1]
        if aux_sampling_method == SiamesePairSamplingMethod.interval:
            begin = x_index - aux_frame_range
            end = x_index + aux_frame_range
            if begin < 0:
                begin = 0
            if end > len(mask):
                end = len(mask)
            masked_candidates = np.arange(begin, end)
            if mask is not None:
                mask_copied = mask[begin: end].copy()
                mask_copied[x_index - begin] = False
                masked_candidates = masked_candidates[mask_copied]
            else:
                masked_candidates = np.delete(masked_candidates, x_index - begin)
            if len(masked_candidates) == 0:
                return indices
        elif aux_sampling_method == SiamesePairSamplingMethod.causal:
            z_index = indices[0]
            if z_index < x_index:
                begin = max(x_index - aux_frame_range, z_index)
                end = x_index
            else:  # z_index > x_index
                begin = x_index
                end = min(z_index, x_index + aux_frame_range, len(mask))
            if begin == end:
                return indices
            masked_candidates = np.arange(begin, end)
            if mask is not None:
                masked_candidates = masked_candidates[mask[begin: end]]
        else:
            raise NotImplementedError(aux_sampling_method)
        aux_index = rng_engine.choice(masked_candidates)
        return indices[0], indices[1], aux_index
    else:
        raise RuntimeError


from ..SiamFC._algo import _gaussian


def _negative_sampling(length, anchor_index, negative_sample_mask, frame_range, rng_engine):
    begin = anchor_index - frame_range
    end = anchor_index + frame_range
    x_axis_begin_value = -begin * 8 / (2 * frame_range + 1) - 4
    x_axis_end_value = (length - 1 - end) * 8 / (2 * frame_range + 1) + 4
    x_axis_values = np.linspace(x_axis_begin_value, x_axis_end_value, length)
    x_axis_values = x_axis_values[negative_sample_mask]
    if len(x_axis_values) == 0:
        return None
    probability = _gaussian(x_axis_values, 0., 5.)
    probability_sum = probability.sum()
    if probability_sum == 0:
        probability = None
    else:
        probability = probability / probability_sum
    candidates = np.arange(0, length)[negative_sample_mask]
    negative_sample_index = rng_engine.choice(candidates, p=probability)
    return negative_sample_index


def do_triplet_sampling_negative_only(length: int, frame_range: int, aux_frame_range: int, mask: np.ndarray=None, rng_engine: np.random.Generator=np.random.default_rng()):
    assert frame_range > 0
    assert aux_frame_range > 0

    z_index = sample_one_positive(length, mask, rng_engine)
    if mask is None or length == 1:
        return (z_index,)

    false_mask = ~mask
    false_mask[z_index] = False

    negative_x_index = _negative_sampling(length, z_index, false_mask, frame_range, rng_engine)
    if negative_x_index is None:
        return (z_index, )
    negative_aug_index = _negative_sampling(length, negative_x_index, false_mask, aux_frame_range, rng_engine)
    if negative_aug_index is None:
        return z_index, negative_x_index

    return z_index, negative_x_index, negative_aug_index
