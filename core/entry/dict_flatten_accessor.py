# A very simple query language for json / yaml
#
#  following Pytorch nn.Module naming convention
#  JsonPath or Json Query is too complex for our use cases.
#
# examples:
#   config = {'a': {'b': {'c': 1}}}
#   get_config(config, 'a.b.c')
#   >>> 1
#   mod_config(config, 'a.b.c', 2)
#   config
#   >>> {'a': {'b': {'c': 2}}}
#   config = {'a': [{'b': {'c': 1}}, {'b': {'c': 2}}]}
#   get_config(config, 'a.0.b.c')
#   >>> 1
#   mod_config(config, 'a.0.b.c', 3)
#   config
#   >>> {'a': [{'b': {'c': 3}}, {'b': {'c': 2}}]}


def mod_config(config, path: str, value):
    paths = path.split('.')
    for sub_path in paths[:-1]:
        config = config[sub_path] if not sub_path.isdigit() else config[int(sub_path)]
    if paths[-1].isdigit():
        config[int(paths[-1])] = value
    else:
        config[paths[-1]] = value


def get_config(config, path: str):
    paths = path.split('.')
    for sub_path in paths:
        config = config[sub_path] if not sub_path.isdigit() else config[int(sub_path)]
    return config
