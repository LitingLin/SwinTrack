class HeadOutputSelector:
    def __init__(self, selection_config):
        self.selection_config = selection_config

    def __call__(self, predicted, label, context):
        predicted = tuple(predicted[name] for name in self.selection_config)
        return predicted, label, context


def build_data_filter(filter_parameter, *_):
    return HeadOutputSelector(filter_parameter['select'])
