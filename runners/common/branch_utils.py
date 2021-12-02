def get_branch_specific_objects(self, mode, object_name):
    object_ = getattr(self, object_name, None)
    if object_ is None:
        return None

    if mode in object_:
        return object_[mode]
    if None in object_:
        return object_[None]
    return None
