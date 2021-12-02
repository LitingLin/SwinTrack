import re


def get_params_dict(model, optimizer_config):
    model_parameters = dict(model.named_parameters())
    base_lr = optimizer_config['lr']
    param_dict = []
    if 'per_parameter' in optimizer_config:
        for per_parameter_rule in optimizer_config['per_parameter']:
            params = []
            regex_matcher = re.compile(per_parameter_rule['name_regex'])
            for model_parameter_name in list(model_parameters.keys()):
                if regex_matcher.search(model_parameter_name) is not None:
                    params.append(model_parameters[model_parameter_name])
                    model_parameters.pop(model_parameter_name)
            assert len(params) > 0, "rule must be effective"
            if 'lr_mult' in per_parameter_rule:
                rule_lr = base_lr * per_parameter_rule['lr_mult']
            else:
                rule_lr = per_parameter_rule['lr']
            param_dict.append({'params': params, 'lr': rule_lr})
    if len(model_parameters) > 0:
        param_dict.insert(0, {'params': list(model_parameters.values())})
    return param_dict
