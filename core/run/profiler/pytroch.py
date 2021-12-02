import torch
from torch.profiler import profile, ProfilerActivity, tensorboard_trace_handler


class PytorchProfiler:
    def __init__(self, output_path, device: str):
        profile_activities = [ProfilerActivity.CPU]
        if 'cuda' in device:
            profile_activities += [ProfilerActivity.CUDA]
        self.profile = profile(activities=profile_activities,
                               on_trace_ready=tensorboard_trace_handler(output_path),
                               record_shapes=True, with_stack=True, with_flops=True)

    def __enter__(self):
        self.profile.__enter__()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.profile.__exit__(exc_type, exc_val, exc_tb)

    def step(self):
        self.profile.step()
