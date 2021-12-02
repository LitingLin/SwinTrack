from core.run.event_dispatcher.register import EventRegister


def build_data_source(data_config, runtime_vars, config, global_synchronized_rng, local_rng, event_register: EventRegister, context: dict, has_training_run: bool):
    data_pipeline_type = data_config['type']
    if data_pipeline_type == 'SiamFC':
        from data.tracking.methods.SiamFC.builders.source import build_siamfc_data_source
        data_pipelines = build_siamfc_data_source(data_config, runtime_vars, config, global_synchronized_rng, local_rng, event_register, context)
    elif data_pipeline_type == 'Sequential':
        from data.tracking.methods.sequential.builders.source import build_sequential_sampling_data_source
        data_pipelines = build_sequential_sampling_data_source(data_config, runtime_vars, config, global_synchronized_rng, local_rng, event_register, context)
    else:
        raise NotImplementedError(data_pipeline_type)
    return data_pipelines


def build_data_bridge(runtime_vars, branch_config, data_config, runner_config, config, event_register, data_context, has_training_run):
    if data_config['type'] == 'Sequential':
        from data.tracking.methods.sequential.builders.additional_pipeline import build_sequential_sampling_additional_data_pipelines
        return build_sequential_sampling_additional_data_pipelines(runtime_vars, branch_config, data_config, runner_config,
                                                     config, event_register, data_context, has_training_run)
    else:
        return None
