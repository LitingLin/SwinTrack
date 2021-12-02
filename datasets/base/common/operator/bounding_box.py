def try_get_int_bounding_box(bounding_box):
    dtype = int
    new_bounding_box = []
    for v in bounding_box:
        if isinstance(v, int):
            new_bounding_box.append(v)
        elif isinstance(v, float):
            if v.is_integer():
                new_bounding_box.append(int(v))
            else:
                new_bounding_box.append(v)
                dtype = float
        else:
            raise RuntimeError(f'invalid dtype {type(v)} in {bounding_box}')
    return new_bounding_box, dtype


def get_bounding_box(object_: dict):
    bounding_box = object_['bounding_box']
    bounding_box_validity = None
    if 'validity' in bounding_box:
        bounding_box_validity = bounding_box['validity']
    return bounding_box['value'], bounding_box_validity


def set_bounding_box_(object_: dict, bounding_box, validity: bool = None, dtype=None, context=None):
    if validity is not None:
        assert isinstance(validity, bool)
    if dtype is None:
        bounding_box, dtype = try_get_int_bounding_box(bounding_box)
    bounding_box = {
        'value': list(bounding_box),
    }
    if validity is not None:
        bounding_box['validity'] = validity
    object_['bounding_box'] = bounding_box
    if context is not None:
        context.set_bounding_box_dtype(dtype)
