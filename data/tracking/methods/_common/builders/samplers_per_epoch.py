def get_samples_per_epoch(datasets, sampling_config, per_frame=False, align=1, drop_last=True):
    if 'samples_per_epoch' in sampling_config:
        samples_per_epoch = sampling_config['samples_per_epoch']
    else:
        if per_frame:
            samples_per_epoch = sum(tuple(sum(tuple(len(sequence) for sequence in dataset)) for dataset in datasets))
        else:
            samples_per_epoch = sum(tuple(len(dataset) for dataset in datasets))
    if 'repeat_times_per_epoch' in sampling_config:
        repeat_times_per_epoch = sampling_config['repeat_times_per_epoch']
        samples_per_epoch = samples_per_epoch * repeat_times_per_epoch

    assert align > 0
    if align != 1:
        if drop_last:
            samples_per_epoch = samples_per_epoch // align * align
        else:
            samples_per_epoch = (samples_per_epoch + align - 1) // align * align

    return samples_per_epoch
