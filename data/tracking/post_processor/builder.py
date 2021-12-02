def build_response_map_post_processor(response_map_parameters, evaluation_config):
    from .response_map import ResponseMapTrackingPostProcessing
    network_post_processor = ResponseMapTrackingPostProcessing(evaluation_config['window_penalty'] > 0,
                                                               response_map_parameters['size'],
                                                               evaluation_config['window_penalty'])
    return network_post_processor


def build_post_processor(network_config: dict, evaluation_config: dict):
    head_config = network_config['head']
    if head_config['output_protocol']['type'] == 'ResponseMap':
        build_fn = build_response_map_post_processor
    else:
        raise NotImplementedError(f'{head_config["output_protocol"]["type"]}')
    response_map_config = head_config['output_protocol']['parameters']['label']
    if 'scales' not in response_map_config:
        network_post_processor = build_fn(response_map_config, evaluation_config)
    else:
        network_post_processors = []
        for response_map_scale_config in response_map_config['scales']:
            network_post_processors.append(build_fn(response_map_scale_config, evaluation_config))
        if evaluation_config['multi_scale']['type'] == 'max_confidence':
            from .multi_scale import MultiScalePostProcessor
            network_post_processor = MultiScalePostProcessor(network_post_processors)
        else:
            raise NotImplementedError(evaluation_config['multi_scale']['type'])

    return network_post_processor
