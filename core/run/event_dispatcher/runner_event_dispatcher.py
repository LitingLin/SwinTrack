class BranchEventDispatcher:
    def __init__(self, epoch_begin_hooks, epoch_end_hooks,
                 iteration_begin_hooks, iteration_end_hooks,
                 status_collectors):
        self.epoch_begin_hooks = epoch_begin_hooks
        self.epoch_end_hooks = epoch_end_hooks
        self.iteration_begin_hooks = iteration_begin_hooks
        self.iteration_end_hooks = iteration_end_hooks
        self.status_collectors = status_collectors

    def iteration_begin(self, is_training):
        if self.iteration_begin_hooks is None:
            return
        for hook in self.iteration_begin_hooks:
            hook.on_iteration_begin(is_training)

    def iteration_end(self, is_training):
        if self.iteration_end_hooks is None:
            return
        for hook in self.iteration_end_hooks:
            hook.on_iteration_end(is_training)

    def epoch_begin(self, epoch):
        if self.epoch_begin_hooks is None:
            return
        for hook in self.epoch_begin_hooks:
            hook.on_epoch_begin(epoch)

    def epoch_end(self, epoch):
        if self.epoch_end_hooks is None:
            return
        for hook in self.epoch_end_hooks:
            hook.on_epoch_end(epoch)

    def collect_status(self):
        if self.status_collectors is None:
            return None
        return {status_collector_name: status_collector.get_status() for status_collector_name, status_collector in self.status_collectors.items()}
