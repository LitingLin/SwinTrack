import torch
import torch.nn as nn


class ConcatenationBasedDecoder(nn.Module):
    def __init__(self, cross_attention_modules,
                 z_untied_pos_enc, x_untied_pos_enc,
                 rpe_bias_table, rpe_index):
        super(ConcatenationBasedDecoder, self).__init__()
        self.layers = nn.ModuleList(cross_attention_modules)
        self.z_untied_pos_enc = z_untied_pos_enc
        self.x_untied_pos_enc = x_untied_pos_enc
        if rpe_index is not None:
            self.register_buffer('rpe_index', rpe_index, False)
        self.rpe_bias_table = rpe_bias_table

    def forward(self, z, x, z_pos, x_pos):
        '''
            Args:
                z (torch.Tensor): (B, L_z, C)
                x (torch.Tensor): (B, L_x, C)
                z_pos (torch.Tensor | None): (1 or B, L_z, C)
                x_pos (torch.Tensor | None): (1 or B, L_x, C)
            Returns:
                torch.Tensor: (B, L_x, C)
        '''
        concatenated_pos_enc = None
        if z_pos is not None:
            concatenated_pos_enc = torch.cat((z_pos, x_pos), dim=1)

        attn_pos_enc = None
        if self.z_untied_pos_enc is not None:
            z_learned_pos_k = self.z_untied_pos_enc()
            x_learned_pos_q, x_learned_pos_k = self.x_untied_pos_enc()
            attn_pos_enc = x_learned_pos_q @ torch.cat((z_learned_pos_k, x_learned_pos_k), dim=1).transpose(-2, -1).unsqueeze(0)

        if self.rpe_bias_table is not None:
            if attn_pos_enc is not None:
                attn_pos_enc = attn_pos_enc + self.rpe_bias_table(self.rpe_index)
            else:
                attn_pos_enc = self.rpe_bias_table(self.rpe_index)

        for cross_attention in self.layers:
            x = cross_attention(x, torch.cat((z, x), dim=1), x_pos, concatenated_pos_enc, attn_pos_enc)
        return x


def build_feature_map_generation_decoder(config, drop_path_allocator,
                                         dim, num_heads, mlp_ratio, qkv_bias, drop_rate, attn_drop_rate,
                                         z_shape, x_shape):
    transformer_config = config['transformer']

    traditional_positional_encoding_enabled = transformer_config['position_embedding']['enabled']

    untied_z_pos_enc = None
    untied_x_pos_enc = None
    rpe_index = None
    rpe_bias_table = None

    untied_position_embedding_config = transformer_config['untied_position_embedding']
    if untied_position_embedding_config['absolute']['enabled']:
        from ...positional_encoding.untied.absolute import Untied2DPositionalEncoder

        untied_z_pos_enc = Untied2DPositionalEncoder(dim, num_heads, z_shape[0], z_shape[1], with_q=False)
        untied_x_pos_enc = Untied2DPositionalEncoder(dim, num_heads, x_shape[0], x_shape[1])

    if untied_position_embedding_config['relative']['enabled']:
        from ...positional_encoding.untied.relative import RelativePosition2DEncoder, generate_2d_concatenated_cross_attention_relative_positional_encoding_index
        rpe_index = generate_2d_concatenated_cross_attention_relative_positional_encoding_index((z_shape[1], z_shape[0]), (x_shape[1], x_shape[0]))
        rpe_bias_table = RelativePosition2DEncoder(num_heads, rpe_index.max() + 1)

    decoder_config = transformer_config['decoder']
    num_layers = decoder_config['num_layers']
    decoder_modules = []

    from ..cross_attention_block import CrossAttentionBlock

    for index_of_decoder in range(num_layers):
        decoder_modules.append(
            CrossAttentionBlock(dim, num_heads, mlp_ratio, qkv_bias, drop=drop_rate, attn_drop=attn_drop_rate,
                                drop_path=drop_path_allocator.allocate(),
                                attn_pos_encoding_only=not traditional_positional_encoding_enabled)
        )
        drop_path_allocator.increase_depth()

    decoder = ConcatenationBasedDecoder(decoder_modules, untied_z_pos_enc, untied_x_pos_enc,  rpe_bias_table, rpe_index)
    return decoder
