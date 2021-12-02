def build_label_generator_and_batch_collator(network_config, with_multi_scale_wrapper=True):
    head_output_protocol = network_config['head']['output_protocol']['type']
    if head_output_protocol == 'ResponseMap':
        from .builders.response_map import build_response_map_label_generator_and_batch_collator
        return build_response_map_label_generator_and_batch_collator(network_config, with_multi_scale_wrapper)
    else:
        raise NotImplementedError(head_output_protocol)
