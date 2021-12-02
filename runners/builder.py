from core.run.event_dispatcher.register import EventRegister


def build_runner(model, runner_config, data_source_context, config, event_register: EventRegister):
    if runner_config['type'] == 'default':
        from .training.default.builder import build_default_training_runner
        return build_default_training_runner(model, runner_config, data_source_context, config, event_register)
    elif runner_config['type'] == 'default_evaluation' or runner_config['type'] == 'coarse_to_fine_evaluation':
        from .evaluation.default import DefaultSiamFCEvaluator
        return DefaultSiamFCEvaluator()
    else:
        raise NotImplementedError(runner_config['type'])
