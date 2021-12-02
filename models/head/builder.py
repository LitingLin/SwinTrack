def build_head(config: dict, with_multi_scale_wrapper=True):
    head_config = config['head']
    head_type = head_config['type']
    if head_type == 'Mlp':
        from .mlp import build_mlp_head
        return build_mlp_head(config, with_multi_scale_wrapper)
    else:
        raise NotImplementedError
