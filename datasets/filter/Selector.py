from ._common import _BaseFilter
from random import Random


class Selector(_BaseFilter):
    def __init__(self, range=None, range_by_ratio=None, indices=None, random=False, random_seed=None):
        paras = {'range': range, 'range_by_ratio': range_by_ratio, 'indices': indices}
        action_name = None
        action_value = None
        for key, value in paras.items():
            if value is not None:
                assert action_name is None
                action_name = key
                action_value = value
        assert action_name is not None
        self.action_name = action_name
        self.action_value = action_value
        self.random_engine = None
        if random:
            self.random_engine = Random(random_seed)

    def __call__(self, length):
        indices = list(range(length))
        if self.random_engine is not None:
            self.random_engine.shuffle(indices)

        if self.action_name == 'range':
            slice_params = {'start': 0, 'stop': length, 'step': 1}
            slice_params.update(self.action_value)
            return indices[slice(slice_params['start'], slice_params['stop'], slice_params['step'])]
        elif self.action_name == 'range_by_ratio':
            slice_params = {'start': 0, 'stop': length, 'step': 1}
            slice_params.update({k: int(round(v*length)) for k, v in self.action_value.items()})
            return indices[slice(slice_params['start'], slice_params['stop'], slice_params['step'])]
        elif self.action_name == 'indices':
            return indices[self.action_value]
        else:
            raise Exception
