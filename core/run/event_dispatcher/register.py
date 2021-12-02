class EventRegister:
    def __init__(self, name_prefix):
        self.started_hooks = None
        self.finished_hooks = None

        self.epoch_begin_hooks = None
        self.epoch_end_hooks = None

        self.stateful_objects = None

        self.status_collectors = None

        self.mode_changed_hooks = None
        self.device_changed_hooks = None
        self.prefix = name_prefix

        self.iteration_begin_hooks = None
        self.iteration_end_hooks = None

    def set_name_prefix(self, prefix):
        self.prefix = prefix

    def register_stateful_object(self, name, object_):
        if self.stateful_objects is None:
            self.stateful_objects = {}
        if self.prefix is not None:
            name = self.prefix + name
        assert name not in self.stateful_objects
        self.stateful_objects[name] = object_

    def register_epoch_begin_hook(self, hook):
        if self.epoch_begin_hooks is None:
            self.epoch_begin_hooks = []
        self.epoch_begin_hooks.append(hook)

    def register_epoch_end_hook(self, hook):
        if self.epoch_end_hooks is None:
            self.epoch_end_hooks = []
        self.epoch_end_hooks.append(hook)

    def register_started_hook(self, hook):
        if self.started_hooks is None:
            self.started_hooks = []
        self.started_hooks.append(hook)

    def register_finished_hook(self, hook):
        if self.finished_hooks is None:
            self.finished_hooks = []
        self.finished_hooks.append(hook)

    def register_status_collector(self, name, object_):
        if self.status_collectors is None:
            self.status_collectors = {}
        if self.prefix is not None:
            name = self.prefix + name
        assert name not in self.status_collectors
        self.status_collectors[name] = object_

    def register_device_changed_hook(self, hook):
        if self.device_changed_hooks is None:
            self.device_changed_hooks = []
        self.device_changed_hooks.append(hook)

    def register_iteration_begin_hook(self, hook):
        if self.iteration_begin_hooks is None:
            self.iteration_begin_hooks = []
        self.iteration_begin_hooks.append(hook)

    def register_iteration_end_hook(self, hook):
        if self.iteration_end_hooks is None:
            self.iteration_end_hooks = []
        self.iteration_end_hooks.append(hook)
