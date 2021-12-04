import torch.nn as nn
from ...self_attention import SelfAttention
from ...cross_attention import CrossAttention
from ...mlp import Mlp


class FeatureFusion(nn.Module):
    def __init__(self,
                 dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=nn.Identity(), act_layer=nn.GELU, norm_layer=nn.LayerNorm, attn_pos_encoding_only=False):
        super(FeatureFusion, self).__init__()
        self.z_norm1 = norm_layer(dim)
        self.x_norm1 = norm_layer(dim)
        self.z_self_attn = SelfAttention(dim, num_heads, qkv_bias, qk_scale, attn_drop, drop, attn_pos_encoding_only)
        self.x_self_attn = SelfAttention(dim, num_heads, qkv_bias, qk_scale, attn_drop, drop, attn_pos_encoding_only)

        self.z_norm2_1 = norm_layer(dim)
        self.z_norm2_2 = norm_layer(dim)
        self.x_norm2_1 = norm_layer(dim)
        self.x_norm2_2 = norm_layer(dim)

        self.z_x_cross_attention = CrossAttention(dim, num_heads, qkv_bias, qk_scale, attn_drop, drop, attn_pos_encoding_only)
        self.x_z_cross_attention = CrossAttention(dim, num_heads, qkv_bias, qk_scale, attn_drop, drop, attn_pos_encoding_only)

        mlp_hidden_dim = int(dim * mlp_ratio)
        self.z_norm3 = norm_layer(dim)
        self.x_norm3 = norm_layer(dim)

        self.z_mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.x_mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        self.drop_path = drop_path

    def forward(self, z, x, z_self_attn_pos, x_self_attn_pos, z_x_cross_attn_pos, x_z_cross_attn_pos):
        z = z + self.drop_path(self.z_self_attn(self.z_norm1(z), None, None, z_self_attn_pos))
        x = x + self.drop_path(self.x_self_attn(self.x_norm1(x), None, None, x_self_attn_pos))

        z = z + self.drop_path(self.z_x_cross_attention(self.z_norm2_1(z), self.x_norm2_1(x), None, None, z_x_cross_attn_pos))
        x = x + self.drop_path(self.x_z_cross_attention(self.x_norm2_2(x), self.z_norm2_2(z), None, None, x_z_cross_attn_pos))

        z = z + self.drop_path(self.z_mlp(self.z_norm3(z)))
        x = x + self.drop_path(self.x_mlp(self.x_norm3(x)))
        return z, x


class FeatureFusionEncoder(nn.Module):
    def __init__(self, feature_fusion_layers, z_pos_enc, x_pos_enc,
                 z_rel_pos_index, x_rel_pos_index, z_x_rel_pos_index, x_z_rel_pos_index,
                 z_rel_pos_bias_table, x_rel_pos_bias_table, z_x_rel_pos_bias_table, x_z_rel_pos_bias_table):
        super(FeatureFusionEncoder, self).__init__()
        self.layers = nn.ModuleList(feature_fusion_layers)
        self.z_pos_enc = z_pos_enc
        self.x_pos_enc = x_pos_enc
        self.register_buffer('z_rel_pos_index', z_rel_pos_index, False)
        self.register_buffer('x_rel_pos_index', x_rel_pos_index, False)
        self.register_buffer('z_x_rel_pos_index', z_x_rel_pos_index, False)
        self.register_buffer('x_z_rel_pos_index', x_z_rel_pos_index, False)
        self.z_rel_pos_bias_table = z_rel_pos_bias_table
        self.x_rel_pos_bias_table = x_rel_pos_bias_table
        self.z_x_rel_pos_bias_table = z_x_rel_pos_bias_table
        self.x_z_rel_pos_bias_table = x_z_rel_pos_bias_table

    def forward(self, z, x, z_pos, x_pos):
        '''
            Args:
                z (torch.Tensor): (B, L_z, C), template image feature tokens
                x (torch.Tensor): (B, L_x, C), search image feature tokens
                z_pos (torch.Tensor | None): (1 or B, L_z, C), optional positional encoding for z
                x_pos (torch.Tensor | None): (1 or B, L_x, C), optional positional encoding for x
            Returns:
                Tuple[torch.Tensor, torch.Tensor]:
                    (B, L_z, C): template image feature tokens
                    (B, L_x, C): search image feature tokens
        '''
        # Support untied positional encoding only for simplicity
        assert z_pos is None and x_pos is None

        # untied positional encoding
        z_q_pos, z_k_pos = self.z_pos_enc()
        x_q_pos, x_k_pos = self.x_pos_enc()
        z_self_attn_pos = (z_q_pos @ z_k_pos.transpose(-2, -1)).unsqueeze(0)
        x_self_attn_pos = (x_q_pos @ x_k_pos.transpose(-2, -1)).unsqueeze(0)

        z_x_cross_attn_pos = (z_q_pos @ x_k_pos.transpose(-2, -1)).unsqueeze(0)
        x_z_cross_attn_pos = (x_q_pos @ z_k_pos.transpose(-2, -1)).unsqueeze(0)

        # relative positional encoding
        z_self_attn_pos = z_self_attn_pos + self.z_rel_pos_bias_table(self.z_rel_pos_index)
        x_self_attn_pos = x_self_attn_pos + self.x_rel_pos_bias_table(self.x_rel_pos_index)
        z_x_cross_attn_pos = z_x_cross_attn_pos + self.z_x_rel_pos_bias_table(self.z_x_rel_pos_index)
        x_z_cross_attn_pos = x_z_cross_attn_pos + self.x_z_rel_pos_bias_table(self.x_z_rel_pos_index)

        for layer in self.layers:
            z, x = layer(z, x, z_self_attn_pos, x_self_attn_pos, z_x_cross_attn_pos, x_z_cross_attn_pos)

        return z, x
