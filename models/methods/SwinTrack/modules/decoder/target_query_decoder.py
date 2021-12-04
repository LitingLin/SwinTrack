import torch
import torch.nn as nn
from ..self_attention import SelfAttention
from ..cross_attention import CrossAttention
from ..mlp import Mlp
from timm.models.layers import trunc_normal_


class TargetQueryDecoderLayer(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None,  drop=0., attn_drop=0.,
                 drop_path=nn.Identity(), act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super(TargetQueryDecoderLayer, self).__init__()
        self.norm_1 = norm_layer(dim)
        self.self_attn = SelfAttention(dim, num_heads, qkv_bias, qk_scale, attn_drop, drop)
        self.norm_2_query = norm_layer(dim)
        self.norm_2_memory = norm_layer(dim)
        self.cross_attn = CrossAttention(dim, num_heads, qkv_bias, qk_scale, attn_drop, drop)
        self.norm_3 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        self.drop_path = drop_path

    def forward(self, query, memory, query_pos, memory_pos):
        '''
            Args:
                query (torch.Tensor): (B, num_queries, C)
                memory (torch.Tensor): (B, L, C)
                query_pos (torch.Tensor): (1 or B, num_queries, C)
                memory_pos (torch.Tensor): (1 or B, L, C)
            Returns:
                torch.Tensor: (B, num_queries, C)
        '''
        query = query + self.drop_path(self.self_attn(self.norm_1(query), query_pos, query_pos, None))
        query = query + self.drop_path(self.cross_attn(self.norm_2_query(query), self.norm_2_memory(memory), query_pos, memory_pos, None))
        query = query + self.drop_path(self.mlp(self.norm_3(query)))

        return query


class TargetQueryDecoderBlock(nn.Module):
    def __init__(self, num_queries, dim, decoder_layers, z_pos_encoder, x_pos_encoder):
        super(TargetQueryDecoderBlock, self).__init__()
        self.target_query = nn.Parameter(torch.empty((num_queries, dim)))
        trunc_normal_(self.target_query, std=.02)
        self.layers = nn.ModuleList(decoder_layers)
        self.z_pos_encoder = z_pos_encoder
        self.x_pos_encoder = x_pos_encoder

    def forward(self, z, x, z_pos=None, x_pos=None):
        '''
            Args:
                z (torch.Tensor): (B, L_z, C)
                x (torch.Tensor): (B, L_x, C)
            Returns:
                torch.Tensor: (B, num_queries, C)
        '''
        assert z_pos is None and x_pos is None
        z_pos = self.z_pos_encoder().unsqueeze(0)
        x_pos = self.x_pos_encoder().unsqueeze(0)
        feat = torch.cat((z, x), dim=1)
        pos = torch.cat((z_pos, x_pos), dim=1)
        N = feat.shape[0]
        target_query = self.target_query.unsqueeze(0).expand(N, -1, -1)
        target = torch.zeros_like(target_query)
        for layer in self.layers:
            target = layer(target, feat, self.target_query, pos)

        return target


def build_target_query_decoder(config, drop_path_allocator,
                               dim, num_heads, mlp_ratio, qkv_bias, drop_rate, attn_drop_rate,
                               z_shape, x_shape):
    decoder_config = config['transformer']['decoder']
    num_layers = decoder_config['num_layers']
    num_queries = decoder_config['num_queries']

    decoder_layers = []
    for _ in range(num_layers):
        decoder_layers.append(
            TargetQueryDecoderLayer(dim, num_heads, mlp_ratio, qkv_bias, drop=drop_rate, attn_drop=attn_drop_rate,
                                    drop_path=drop_path_allocator.allocate()))
        drop_path_allocator.increase_depth()

    from ...positional_encoding.builder import build_position_embedding

    z_positional_encoder, x_positional_encoder = build_position_embedding(decoder_config['positional_encoding'], z_shape, x_shape, dim)

    decoder = TargetQueryDecoderBlock(num_queries, dim, decoder_layers, z_positional_encoder, x_positional_encoder)
    return decoder
