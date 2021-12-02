import os.path

from .distributor import SequenceDistributor
from .test_only import TestOnlyDatasetTrackingResultSaver, Got10kFormatPacker, TrackingNetFormatPacker
from .full_annotated import FullyAnnotatedDatasetReportGenerator


class _OnTheFlyConstructor:
    def __init__(self, config):
        self.config = config

    def construct(self, datasets, saving_path, tracker_name, epoch):
        if saving_path is not None and epoch is not None:
            saving_path = os.path.join(saving_path, f'epoch{epoch}')
        regex_rule_list = []
        collectors = []
        for handler_config in self.config:
            if handler_config['type'] == 'test_only':
                regex_rule_list.append(handler_config['name_regex'])
                packing_path = handler_config['packer']['path']
                format = handler_config['packer']['format']

                packer = None
                if saving_path is not None:
                    packing_path = os.path.join(saving_path, packing_path)
                    if format == 'got10k':
                        packer = Got10kFormatPacker(tracker_name, os.path.join(packing_path, 'submit'))
                    elif format == 'trackingnet':
                        packer = TrackingNetFormatPacker(tracker_name, os.path.join(packing_path, 'submit'))
                    else:
                        raise NotImplementedError(format)

                collectors.append(TestOnlyDatasetTrackingResultSaver(tracker_name, datasets, packing_path, packer))
            elif handler_config['type'] == 'standard':
                if 'name_regex' in handler_config:
                    regex_rule_list.append(handler_config['name_regex'])
                else:
                    regex_rule_list.append('.*')
                collectors.append(FullyAnnotatedDatasetReportGenerator(datasets, tracker_name, saving_path))
            else:
                raise NotImplementedError(handler_config['type'])
        return SequenceDistributor(regex_rule_list), collectors
