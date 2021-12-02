import torch
import torch.nn as nn
from timm.models.layers import trunc_normal_


class Learned2DPositionalEncoder(nn.Module):
    def __init__(self, dim, w, h):
        super(Learned2DPositionalEncoder, self).__init__()
        self.w_pos = nn.Parameter(torch.empty(w, dim))
        self.h_pos = nn.Parameter(torch.empty(h, dim))
        trunc_normal_(self.w_pos, std=0.02)
        trunc_normal_(self.h_pos, std=0.02)

    def forward(self):
        w = self.w_pos.shape[0]
        h = self.h_pos.shape[0]
        return (self.w_pos[None, :, :] + self.h_pos[:, None, :]).view(h * w, -1)
