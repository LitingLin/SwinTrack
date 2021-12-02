import torch.nn as nn


class MultiScaleHead(nn.Module):
    def __init__(self, modules):
        super(MultiScaleHead, self).__init__()
        self.heads = nn.ModuleList(modules)

    def forward(self, inputs):
        return [head(input_) for head, input_ in zip(self.heads, inputs)]
