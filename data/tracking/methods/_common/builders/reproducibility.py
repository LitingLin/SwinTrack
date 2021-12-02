def get_reproducibility_parameters(sampling_config):
    rng_engine_seed = None
    reset_per_epoch = False
    if 'randomness_controlling' in sampling_config:
        rng_config = sampling_config['randomness_controlling']['RNG']
        if 'fixed_seed' in rng_config:
            rng_engine_seed = rng_config['fixed_seed']

        if 'reset_per_epoch' in rng_config and rng_config['reset_per_epoch']:
            reset_per_epoch = True
    return rng_engine_seed, reset_per_epoch
