def nullable_get(object_, subscripts):
    for subscript in subscripts:
        try:
            object_ = object_[subscript]
        except (KeyError, IndexError):
            return None
    return object_
