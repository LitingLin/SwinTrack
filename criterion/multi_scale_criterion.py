import torch.nn as nn


class MultiScaleCriterion(nn.Module):
    def __init__(self, global_data_filter, scale_dispatcher, single_scale_criterion):
        super(MultiScaleCriterion, self).__init__()
        self.global_data_filter = global_data_filter
        self.scale_dispatcher = scale_dispatcher
        self.single_scale_criterion = nn.ModuleList(single_scale_criterion)

    def forward(self, predicted, label):
        if self.global_data_filter is not None:
            predicted, label = self.global_data_filter(predicted, label)
        losses = []
        for index, criterion in enumerate(self.single_scale_criterion):
            losses.append(criterion(*self.scale_dispatcher(predicted, label, index)))
        return losses
