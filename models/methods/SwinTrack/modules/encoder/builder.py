def build_encoder(config, drop_path_allocator,
                  dim, num_heads, mlp_ratio, qkv_bias, drop_rate, attn_drop_rate,
                  z_shape, x_shape):
    encoder_config = config['transformer']['encoder']
    encoder_type = encoder_config['type']

    if encoder_type == 'concatenation_feature_fusion':
        from .concatenated_fusion.builder import build_concatenated_fusion_encoder
        return build_concatenated_fusion_encoder(config, drop_path_allocator,
                                                 dim, num_heads, mlp_ratio, qkv_bias, drop_rate, attn_drop_rate,
                                                 z_shape, x_shape)
    elif encoder_type == 'cross_attention_feature_fusion':
        from .cross_attention_fusion.builder import build_cross_attention_based_encoder
        return build_cross_attention_based_encoder(config, drop_path_allocator,
                                                   dim, num_heads, mlp_ratio, qkv_bias, drop_rate, attn_drop_rate,
                                                   z_shape, x_shape)
    else:
        raise NotImplementedError(encoder_type)
