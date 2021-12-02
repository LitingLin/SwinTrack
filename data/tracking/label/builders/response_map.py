from ..label_generation import build_label_generator_and_batch_collator


def build_response_map_label_generator_and_batch_collator(network_config, with_multi_scale_wrapper=True):
    data_bounding_box_parameters = network_config['data']['bounding_box']
    assert data_bounding_box_parameters['format'] == 'CXCYWH'
    assert data_bounding_box_parameters['normalization_protocol']['interval'] == '[)'
    assert data_bounding_box_parameters['normalization_protocol']['range'] == [0, 1]

    return build_label_generator_and_batch_collator(network_config, True, with_multi_scale_wrapper)
