def string_to_number(s: str):
    try:
        return int(s)
    except ValueError:
        return float(s)


def string_to_number_slow(s: str):
    try:
        return int(s)
    except ValueError:
        v = float(s)
        i_v = int(v)
        if v == i_v:
            return i_v
        else:
            return v
