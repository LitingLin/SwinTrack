class ConstantWarmupScheduler:
    def __init__(self, warmup_steps, warmup_value, ultimate_value, per_iteration):
        self.warmup_steps = warmup_steps
        self.warmup_value = warmup_value
        self.ultimate_value = ultimate_value
        self.per_iteration = per_iteration
        self.position = 0

    def state_dict(self):
        return self.position

    def load_state_dict(self, state):
        self.position = state

    def on_iteration_end(self, is_training):
        if is_training:
            if self.per_iteration:
                self.position += 1

    def on_epoch_begin(self, epoch):
        if not self.per_iteration:
            self.position = epoch

    def current(self):
        if self.position < self.warmup_steps:
            return self.warmup_value
        else:
            return self.ultimate_value
