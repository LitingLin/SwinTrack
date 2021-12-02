import torch

from miscellanies.torch.distributed import is_dist_available_and_initialized
from miscellanies.torch.distributed.reduce_mean import reduce_mean_
from models.utils.scheduler.linear_scheduler import LinearScheduler


class LinearWeightScheduler:
    def __init__(self, init_weight, ultimate_weight, begin_step, end_step, per_iter):
        self.linear_scheduler = LinearScheduler(init_weight, ultimate_weight, begin_step, end_step, per_iter)

    def state_dict(self):
        return self.linear_scheduler.state_dict()

    def load_state_dict(self, state):
        self.linear_scheduler.load_state_dict(state)

    def forward(self, loss):
        weight = self.linear_scheduler.current()
        return loss * weight, loss.detach(), weight

    def on_iteration_end(self, is_training):
        self.linear_scheduler.on_iteration_end(is_training)

    def on_epoch_begin(self, epoch):
        self.linear_scheduler.on_epoch_begin(epoch)


class ConstantWeightScheduler:
    def __init__(self, weight):
        self.weight = weight

    def forward(self, loss):
        return loss * self.weight, loss.detach(), self.weight


def _get_display_name(display_name, scaled, prefix, postfix):
    if not scaled:
        display_name += '_unscaled'
    if prefix is not None:
        display_name = prefix + display_name
    if postfix is not None:
        display_name += postfix

    return display_name


class LossComposer:
    def __init__(self, loss_weight_schedulers, display_names, display_prefix, display_postfix):
        self.loss_weight_schedulers = loss_weight_schedulers
        self.display_names = display_names
        self.display_prefix = display_prefix
        self.display_postfix = display_postfix

    def __call__(self, losses):
        loss_list = []
        detached_loss_list = []
        weight_list = []
        for index, loss_weight_scheduler in enumerate(self.loss_weight_schedulers):
            loss, loss_detached, weight = loss_weight_scheduler.forward(losses[index])
            loss_list.append(loss)
            detached_loss_list.append(loss_detached)
            weight_list.append(weight)

        weighted_loss = sum(loss_list)
        if is_dist_available_and_initialized():
            detached_loss_list = torch.stack(detached_loss_list, dim=0)
            reduce_mean_(detached_loss_list)
            detached_loss_list = detached_loss_list.cpu()

        loss_stats_dict = {}
        unscaled_loss_stats_dict = {}

        for detached_loss, weight, display_name in zip(detached_loss_list, weight_list, self.display_names):
            detached_loss = detached_loss.cpu().item()
            loss_stats_dict[_get_display_name(display_name, True, self.display_prefix, self.display_postfix)] = detached_loss * weight
            unscaled_loss_stats_dict[_get_display_name(display_name, False, self.display_prefix, self.display_postfix)] = detached_loss

        loss_value = sum(loss_stats_dict.values())
        loss_stats_dict.update(unscaled_loss_stats_dict)

        return weighted_loss, loss_value, loss_stats_dict

    def state_dict(self):
        state = []
        for loss_weight_scheduler in self.loss_weight_schedulers:
            if hasattr(loss_weight_scheduler, 'state_dict'):
                state.append(loss_weight_scheduler.state_dict())
        return state

    def load_state_dict(self, state):
        pos = 0
        for loss_weight_scheduler in self.loss_weight_schedulers:
            if hasattr(loss_weight_scheduler, 'load_state_dict'):
                loss_weight_scheduler.load_state_dict(state[pos])
                pos += 1

    def on_iteration_end(self, is_training):
        for loss_weight_scheduler in self.loss_weight_schedulers:
            if hasattr(loss_weight_scheduler, 'on_iteration_end'):
                loss_weight_scheduler.on_iteration_end(is_training)

    def on_epoch_begin(self, epoch):
        for loss_weight_scheduler in self.loss_weight_schedulers:
            if hasattr(loss_weight_scheduler, 'on_epoch_begin'):
                loss_weight_scheduler.on_epoch_begin(epoch)
