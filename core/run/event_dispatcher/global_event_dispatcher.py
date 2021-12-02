class GlobalEventDispatcher:
    def __init__(self, started_hooks, finished_hooks, stateful_objects, device_changed_hooks):
        self.stateful_objects = stateful_objects

        self.started_hooks = started_hooks
        self.finished_hooks = finished_hooks

        self.device_changed_hooks = device_changed_hooks

    def device_changed(self, device):
        if self.device_changed_hooks is not None:
            for hook in self.device_changed_hooks:
                hook.on_device_changed(device)

    def __enter__(self):
        self.start()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()

    def start(self):
        if self.started_hooks is not None:
            for hook in self.started_hooks:
                hook.on_started()

    def stop(self):
        if self.finished_hooks is not None:
            for hook in self.finished_hooks:
                hook.on_finished()

    def collect_state_dict(self):
        if self.stateful_objects is None:
            return None
        object_states = {}
        for state_name, stateful_object in self.stateful_objects.items():
            object_states[state_name] = stateful_object.state_dict()
        return object_states

    def dispatch_state_dict(self, object_states):
        if self.stateful_objects is not None:
            for state_name, stateful_object in self.stateful_objects.items():
                stateful_object.load_state_dict(object_states[state_name])
