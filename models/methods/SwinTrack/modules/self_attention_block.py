import torch.nn as nn
from .self_attention import PVTSelfAttention
from .mlp import Mlp


class SelfAttentionBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None,  drop=0., attn_drop=0.,
                 drop_path=nn.Identity(), act_layer=nn.GELU, norm_layer=nn.LayerNorm, attn_pos_encoding_only=False):
        super(SelfAttentionBlock, self).__init__()
        self.norm1 = norm_layer(dim)
        self.attn = PVTSelfAttention(dim, num_heads, qkv_bias, qk_scale, attn_drop, drop, attn_pos_encoding_only)

        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = drop_path
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, q_ape, k_ape, attn_ape):
        x = x + self.drop_path(self.attn(self.norm1(x), q_ape, k_ape, attn_ape))
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x
