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
