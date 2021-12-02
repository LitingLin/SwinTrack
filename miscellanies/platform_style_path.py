import os


def get_platform_style_path(path):
    if os.name == 'nt':
        path = path.replace('/', '\\')
    else:
        path = path.replace('\\', '/')
    return path


def join_path(*args):
    path = os.path.abspath(os.path.join(*args))
    if os.name == 'nt':
        path = path.replace('/', '\\')
    else:
        path = path.replace('\\', '/')
    return path
