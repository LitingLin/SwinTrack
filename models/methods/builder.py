from core.run.event_dispatcher.register import EventRegister


def build_model(config: dict, runtime_vars, max_batch_size, num_epochs: int, iterations_per_epoch: int, event_register: EventRegister, has_training_run: bool):
    load_pretrain = runtime_vars.resume is None and runtime_vars.weight_path is None
    if config['version'] == 1 and config['type'] == 'SwinTrack':
        from .SwinTrack.builder import build_swin_track
        return build_swin_track(config, load_pretrain, num_epochs, iterations_per_epoch, event_register, has_training_run)
    else:
        raise NotImplementedError(f'Unknown version {config["version"]}')
