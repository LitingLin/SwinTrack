import torch.nn as nn
from core.operator.sine_position_encoding import generate_2d_sine_position_encoding, generate_2d_sine_position_encoding_with_index


def generate_transformer_2d_sine_positional_encoding(h, w, dim):
    encoding = generate_2d_sine_position_encoding(1, h, w, dim)  # (1, L, C)
    return encoding.view(h * w, dim)


def generate_transformer_2d_sine_positional_encoding_with_index(index, h, w, dim):
    encoding = generate_2d_sine_position_encoding_with_index(index, 1, h, w, dim)  # (1, L, C)
    return encoding.view(h * w, dim)


class SinePositionEmbedding(nn.Module):
    def __init__(self, dim, shape, index=None):
        super(SinePositionEmbedding, self).__init__()
        if index is None:
            self.register_buffer('positional_encoding',
                                 generate_transformer_2d_sine_positional_encoding(shape[0], shape[1], dim), False)
        else:
            self.register_buffer('positional_encoding',
                                 generate_transformer_2d_sine_positional_encoding_with_index(index, shape[0], shape[1], dim), False)

    def forward(self):
        return self.positional_encoding
