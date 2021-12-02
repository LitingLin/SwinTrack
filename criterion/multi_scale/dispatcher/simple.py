def simple_scale_dispatcher(predicted, label, index):
    return predicted[index], label[index]


def build_multi_scale_data_dispatcher(*_):
    return simple_scale_dispatcher
