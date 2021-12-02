import os


def get_repository_root():
    root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    return root_path
