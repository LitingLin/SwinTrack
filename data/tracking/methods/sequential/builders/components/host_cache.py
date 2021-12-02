from ...pipeline.host_cache import UUIDBasedCacheService


def build_host_cache(config, max_cache_length):
    if config['type'] == 'token':
        from ...pipeline.host_cache import TokenCache
        return TokenCache(max_cache_length, config['dim'], config['length'])
    elif config['type'] == 'feature_map':
        from ...pipeline.host_cache import FeatureMapCache
        return FeatureMapCache(max_cache_length, config['dim'], config['shape'])
    elif config['type'] == 'scalar':
        from ...pipeline.host_cache import ScalerCache
        return ScalerCache(max_cache_length, config['dim'])
    elif config['type'] == 'multi_scale_token':
        from ...pipeline.host_cache import MultiScaleTokenCache
        return MultiScaleTokenCache(max_cache_length, config['dim'], config['length'])
    else:
        raise NotImplementedError(config['type'])


def build_tracking_procedure_template_cache(runtime_vars, branch_config, batch_size):
    num_workers = runtime_vars.num_workers
    if num_workers == 0:
        num_workers = 1
    max_cache_size = num_workers * batch_size
    template_feature_cache_config = branch_config['tracking']['cache']['template']
    template_image_mean_cache_config = branch_config['tracking']['cache']['image_mean']
    return UUIDBasedCacheService(max_cache_size, build_host_cache(template_feature_cache_config, max_cache_size)), \
           UUIDBasedCacheService(max_cache_size, build_host_cache(template_image_mean_cache_config, max_cache_size))
