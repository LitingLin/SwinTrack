from .dict_flatten_accessor import mod_config, get_config
import os
from miscellanies.yaml_ops import load_yaml


def _apply_mixin_rule(rule: dict, config, value, action=None):
    query_path = rule['path']

    # 'replace' action is the default action
    if action is None:
        if 'action' not in rule:
            action = 'replace'
        else:
            action = rule['action']

    if value is None:
        # fixed static rule
        value = rule['value']

    if action == 'replace':  # replace the config in path by given value
        if isinstance(query_path, (list, tuple)):  # multiple paths
            [mod_config(config, sub_access_path, value) for sub_access_path in query_path]
        else:
            mod_config(config, query_path, value)
    elif action == 'insert':
        if isinstance(query_path, (list, tuple)):
            for sub_access_path in query_path:
                config = get_config(config, sub_access_path)
                if isinstance(value, (list, tuple)):
                    config.extend(value)
                elif isinstance(value, dict):
                    config.update(value)
                else:
                    config.insert(value)
        else:
            config = get_config(config, query_path)
            if isinstance(value, (list, tuple)):
                config.extend(value)
            elif isinstance(value, dict):
                config.update(value)
            else:
                config.insert(value)
    elif action == 'include':
        if isinstance(query_path, (list, tuple)):
            for sub_access_path in query_path:
                config = get_config(config, sub_access_path)
                assert isinstance(config, dict)
                for key in list(config.keys()):
                    if key != value:
                        del config[key]
        else:
            config = get_config(config, query_path)
            assert isinstance(config, dict)
            for key in list(config.keys()):
                if key != value:
                    del config[key]
    else:
        raise NotImplementedError(action)


def apply_mixin_rules(mixin_rules, config, dynamic_values):
    if 'fixed' in mixin_rules:
        for fixed_modification_rule in mixin_rules['fixed']:
            _apply_mixin_rule(fixed_modification_rule, config, None)
    if 'dynamic' in mixin_rules:
        for dynamic_parameter_name, dynamic_modification_rule in mixin_rules['dynamic'].items():
            _apply_mixin_rule(dynamic_modification_rule, config, dynamic_values[dynamic_parameter_name])


def get_mixin_config(args):
    configs = []
    for mixin_config in args.mixin_config:
        if mixin_config.startswith('/' or '\\'):
            config_path = os.path.join(args.config_path, mixin_config[1:])
        else:
            config_path = os.path.join(args.config_path, args.method_name, args.config_name, 'mixin', mixin_config)
            if not os.path.exists(config_path):
                config_path = os.path.join(args.config_path, 'mixin', mixin_config)
        assert os.path.exists(config_path), 'Mixin config not found: {}'.format(config_path)
        configs.append(load_yaml(config_path))
    return configs


def load_static_mixin_config_and_apply_rules(args, config):
    mixin_configs = get_mixin_config(args)
    for mixin_config in mixin_configs:
        apply_mixin_rules(mixin_config['mixin'], config, None)
