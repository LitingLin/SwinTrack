import torch
from .params import get_params_dict


def build_optimizer(model, optimizer_config: dict):
    optimizer_params = get_params_dict(model, optimizer_config)
    lr = optimizer_config['lr']
    weight_decay = optimizer_config['weight_decay']

    if optimizer_config['type'] == 'SGD':
        momentum = optimizer_config['momentum']
        nesterov = optimizer_config['nesterov']
        optimizer = torch.optim.SGD(optimizer_params, lr=lr, weight_decay=weight_decay, nesterov=nesterov,
                                    momentum=momentum)
    elif optimizer_config['type'] == 'AdamW':
        optimizer = torch.optim.AdamW(optimizer_params, lr=lr,
                                      weight_decay=weight_decay)
    else:
        raise NotImplementedError(f'Unknown lr_scheduler {optimizer_config["type"]}')

    return optimizer
