from miscellanies.yaml_ops import load_yaml
import os
_cached_config = None


def get_path_from_config(name: str):
    global _cached_config
    if _cached_config is None:
        config_folder = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
        config_path = os.path.join(config_folder, 'path.yaml')
        if not os.path.exists(config_path):
            import shutil
            shutil.copyfile(os.path.join(config_folder, 'path.template.yaml'), config_path)
            raise RuntimeError('Setup the paths in path.yaml first')
        _cached_config = load_yaml(config_path)
    return _cached_config[name]
