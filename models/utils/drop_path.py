from typing import List
import torch
from torch.nn import Identity
from timm.models.layers import DropPath


class DropPathAllocator:
    def __init__(self, max_drop_path_rate, stochastic_depth_decay = True):
        self.max_drop_path_rate = max_drop_path_rate
        self.stochastic_depth_decay = stochastic_depth_decay
        self.allocated = []
        self.allocating = None

    def __enter__(self):
        self.allocating = []

    def __exit__(self, exc_type, exc_val, exc_tb):
        if len(self.allocating) != 0:
            self.allocated.append(self.allocating)
        self.allocating = None
        if not self.stochastic_depth_decay:
            for depth_module in self.allocated:
                for module in depth_module:
                    if isinstance(module, DropPath):
                        module.drop_prob = self.max_drop_path_rate
        else:
            depth = self.get_depth()
            dpr = [x.item() for x in torch.linspace(0, self.max_drop_path_rate, depth)]
            assert len(dpr) == len(self.allocated)
            for drop_path_rate, depth_modules in zip(dpr, self.allocated):
                for module in depth_modules:
                    if isinstance(module, DropPath):
                        module.drop_prob = drop_path_rate

    def __len__(self):
        length = 0

        for depth_modules in self.allocated:
            length += len(depth_modules)

        return length

    def increase_depth(self):
        self.allocated.append(self.allocating)
        self.allocating = []

    def get_depth(self):
        return len(self.allocated)

    def allocate(self):
        if self.max_drop_path_rate == 0 or (self.stochastic_depth_decay and self.get_depth() == 0):
            drop_path_module = Identity()
        else:
            drop_path_module = DropPath()
        self.allocating.append(drop_path_module)
        return drop_path_module

    def get_all_allocated(self):
        allocated = []
        for depth_module in self.allocated:
            for module in depth_module:
                allocated.append(module)
        return allocated


class DropPathScheduler:
    def __init__(self, drop_path_modules: List[DropPath], scheduler):
        self.modules = [m for m in drop_path_modules if isinstance(m, DropPath)]
        self.module_initial_drop_probs = [m.drop_prob for m in drop_path_modules if isinstance(m, DropPath)]
        self.scheduler = scheduler
        self._update_all()

    def state_dict(self):
        return self.scheduler.get_state(), self.module_initial_drop_probs

    def load_state_dict(self, state):
        assert len(state) == 2
        self.scheduler.set_state(state[0])
        self.module_initial_drop_probs = state[1]
        assert len(self.module_initial_drop_probs) == len(self.modules)
        self._update_all()

    def on_iteration_end(self, is_training):
        self.scheduler.on_iteration_end(is_training)
        if self.scheduler.per_iteration:
            self._update_all()

    def on_epoch_begin(self, epoch):
        self.scheduler.on_epoch_begin(epoch)

        if not self.scheduler.per_iteration:
            self._update_all()

    def _update_all(self):
        ratio = self.scheduler.current()

        for module, origin_drop_rate in zip(self.modules, self.module_initial_drop_probs):
            module.drop_prob = origin_drop_rate * ratio
