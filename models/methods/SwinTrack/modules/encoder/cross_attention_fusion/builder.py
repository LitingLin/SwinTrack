from .cross_attention_fusion import FeatureFusion, FeatureFusionEncoder
from ....positional_encoding.untied.absolute import Untied2DPositionalEncoder
from ....positional_encoding.untied.relative import generate_2d_relative_positional_encoding_index, \
    RelativePosition2DEncoder


def build_cross_attention_based_encoder(config: dict, drop_path_allocator,
                                        dim, num_heads, mlp_ratio, qkv_bias, drop_rate, attn_drop_rate,
                                        z_shape, x_shape):
    transformer_config = config['transformer']

    assert transformer_config['position_embedding']['enabled'] == False and \
           transformer_config['untied_position_embedding']['absolute']['enabled'] == True and \
           transformer_config['untied_position_embedding']['relative']['enabled'] == True

    encoder_config = transformer_config['encoder']
    assert encoder_config['type'] == 'cross_attention_feature_fusion'

    num_layers = encoder_config['num_layers']

    encoder_layers = []
    for index_of_layer in range(num_layers):
        encoder_layers.append(
            FeatureFusion(dim, num_heads, mlp_ratio, qkv_bias, drop=drop_rate, attn_drop=attn_drop_rate,
                          drop_path=drop_path_allocator.allocate(),
                          attn_pos_encoding_only=True)
        )

    z_abs_encoder = Untied2DPositionalEncoder(dim, num_heads, z_shape[0], z_shape[1])
    x_abs_encoder = Untied2DPositionalEncoder(dim, num_heads, x_shape[0], x_shape[1])

    z_self_attn_rel_pos_index = generate_2d_relative_positional_encoding_index(z_shape, z_shape)
    x_self_attn_rel_pos_index = generate_2d_relative_positional_encoding_index(x_shape, x_shape)

    z_x_cross_attn_rel_pos_index = generate_2d_relative_positional_encoding_index(z_shape, x_shape)
    x_z_cross_attn_rel_pos_index = generate_2d_relative_positional_encoding_index(x_shape, z_shape)

    z_self_attn_rel_pos_bias_table = RelativePosition2DEncoder(num_heads, z_self_attn_rel_pos_index.max() + 1)
    x_self_attn_rel_pos_bias_table = RelativePosition2DEncoder(num_heads, x_self_attn_rel_pos_index.max() + 1)
    z_x_cross_attn_rel_pos_bias_table = RelativePosition2DEncoder(num_heads, z_x_cross_attn_rel_pos_index.max() + 1)
    x_z_cross_attn_rel_pos_bias_table = RelativePosition2DEncoder(num_heads, x_z_cross_attn_rel_pos_index.max() + 1)

    return FeatureFusionEncoder(encoder_layers, z_abs_encoder, x_abs_encoder, z_self_attn_rel_pos_index,
                                x_self_attn_rel_pos_index,
                                z_x_cross_attn_rel_pos_index, x_z_cross_attn_rel_pos_index,
                                z_self_attn_rel_pos_bias_table,
                                x_self_attn_rel_pos_bias_table, z_x_cross_attn_rel_pos_bias_table,
                                x_z_cross_attn_rel_pos_bias_table)
