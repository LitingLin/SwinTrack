import yaml
from yaml import CSafeLoader as Loader, CSafeDumper as Dumper
import os


__all__ = ['load_yaml', 'dump_yaml']


class CustomLoader(Loader):
    def __init__(self, stream):
        self._root = os.path.split(stream.name)[0]

        super(CustomLoader, self).__init__(stream)

    def include(self, node):
        filename = os.path.join(self._root, self.construct_scalar(node))

        if os.name == 'nt':
            filename = filename.replace('/', '\\')
        with open(filename, 'r') as f:
            return yaml.load(f, CustomLoader)


CustomLoader.add_constructor('!include', CustomLoader.include)


def load_yaml(path: str, loader=CustomLoader):
    with open(path, 'rb') as f:
        object_ = yaml.load(f, Loader=loader)
    return object_


def dump_yaml(object_, path: str, dumper=Dumper):
    with open(path, 'wb') as f:
        yaml.dump(object_, f, encoding='utf-8', default_flow_style=False, Dumper=dumper)
