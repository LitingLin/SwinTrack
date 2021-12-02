from ...curation_parameter_provider import SiamFCCurationParameterSimpleProvider


def build_curation_parameter_provider(evaluation_config, default_area_factor):
    tracking_curation_process_method_config = evaluation_config['curation_parameter_provider']

    tracking_curation_process_method = tracking_curation_process_method_config['type']
    min_object_size = tracking_curation_process_method_config['min_object_size']
    if isinstance(min_object_size, (int, float)):
        min_object_size = (min_object_size, min_object_size)

    if tracking_curation_process_method == 'simple':
        return SiamFCCurationParameterSimpleProvider(default_area_factor, min_object_size)
    else:
        raise NotImplementedError(tracking_curation_process_method)
