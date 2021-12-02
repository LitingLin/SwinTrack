import configparser


def parse_ini_file(path: str):
    parser = configparser.ConfigParser()
    parser.read(path)
    return {s: dict(parser.items(s)) for s in parser.sections()}
