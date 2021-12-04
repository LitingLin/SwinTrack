import torch
import torch.nn as nn


class ConcatenatedFusion(nn.Module):
    def __init__(self, concatenated_self_attention_modules,
                 z_untied_pos_enc, x_untied_pos_enc,
                 rpe_bias_table, rpe_index):
        super(ConcatenatedFusion, self).__init__()
        self.layers = nn.ModuleList(concatenated_self_attention_modules)
        self.z_untied_pos_enc = z_untied_pos_enc
        self.x_untied_pos_enc = x_untied_pos_enc
        if rpe_index is not None:
            self.register_buffer('rpe_index', rpe_index, False)
        self.rpe_bias_table = rpe_bias_table

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
        concatenated = torch.cat((z, x), dim=1)

        attn_pos_enc = None
        if self.z_untied_pos_enc is not None:
            z_q_pos, z_k_pos = self.z_untied_pos_enc()
            x_q_pos, x_k_pos = self.x_untied_pos_enc()
            attn_pos_enc = (torch.cat((z_q_pos, x_q_pos), dim=1) @ torch.cat((z_k_pos, x_k_pos), dim=1).transpose(-2, -1)).unsqueeze(0)

        if self.rpe_bias_table is not None:
            if attn_pos_enc is not None:
                attn_pos_enc = attn_pos_enc + self.rpe_bias_table(self.rpe_index)
            else:
                attn_pos_enc = self.rpe_bias_table(self.rpe_index)
        concatenated_pos_enc = None
        if z_pos is not None:
            assert x_pos is not None
            concatenated_pos_enc = torch.cat((z_pos, x_pos), dim=1)
        for merged_self_attention in self.layers:
            concatenated = merged_self_attention(concatenated, concatenated_pos_enc, concatenated_pos_enc, attn_pos_enc)
        return concatenated[:, :z.shape[1], :], concatenated[:, z.shape[1]:, :]
