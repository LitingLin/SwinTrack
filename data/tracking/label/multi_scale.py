from miscellanies.collate import collate_batch_list


class SimpleMultiScaleLabelGeneratorWrapper:
    def __init__(self, label_generators):
        self.label_generators = label_generators

    def __call__(self, *args):
        return tuple(label_generator(*args) for label_generator in self.label_generators)


class SimpleMultiScaleLabelBatchCollator:
    def __init__(self, single_scale_label_batch_collator):
        self.single_scale_label_batch_collator = single_scale_label_batch_collator

    def __call__(self, label_list):
        multi_scale_labels = collate_batch_list(label_list)
        labels = tuple(self.single_scale_label_batch_collator(single_scale_label) for single_scale_label in multi_scale_labels)
        return labels
