from runners.common.branch_utils import get_branch_specific_objects
from runners.interface import BaseRunner


def _run_fn(fn, args):
    if isinstance(args, (list, tuple)):
        return fn(*args)
    elif isinstance(args, dict):
        return fn(**args)
    else:
        return fn(args)


class DefaultSiamFCEvaluator(BaseRunner):
    def __init__(self):
        self.data_pipeline_on_host = None
        self.tracker_evaluator = None

        self.branch_name = None

    def switch_branch(self, branch_name):
        self.branch_name = branch_name

    def train(self, is_train):
        assert not is_train, "Evaluator can only be run in evaluation mode"

    def get_iteration_index(self):
        return None

    def get_metric_definitions(self):
        metric_definitions = []
        data_pipelines = get_branch_specific_objects(self, self.branch_name, 'data_pipeline_on_host')
        for data_pipeline in data_pipelines:
            if hasattr(data_pipeline, 'get_metric_definitions'):
                metric_definitions.append(data_pipeline.get_metric_definitions())
        if self.tracker_evaluator is not None:
            if hasattr(self.tracker_evaluator, 'get_metric_definitions'):
                metric_definitions.append(self.tracker_evaluator.get_metric_definitions())
        if len(metric_definitions) == 0:
            metric_definitions = None

        return metric_definitions

    def run_iteration(self, model, data):
        samples, targets, miscellanies_on_host, miscellanies_on_device = data
        assert self.branch_name is not None
        data_pipeline_on_host = get_branch_specific_objects(self, self.branch_name, 'data_pipeline_on_host')
        tracker_evaluator = get_branch_specific_objects(self, self.branch_name, 'tracker_evaluator')

        if data_pipeline_on_host is not None:
            for data_pipeline in data_pipeline_on_host:
                if hasattr(data_pipeline, 'pre_processing'):
                    samples, targets, miscellanies_on_host, miscellanies_on_device = data_pipeline.pre_processing(samples, targets, miscellanies_on_host, miscellanies_on_device)

        outputs = None
        if tracker_evaluator is None:
            if samples is not None:
                outputs = _run_fn(samples, samples)
        else:
            initialization_samples = tracker_evaluator.pre_initialization(
                samples, targets, miscellanies_on_host, miscellanies_on_device)
            tracker_initialization_results = None
            if initialization_samples is not None:
                tracker_initialization_results = _run_fn(model.initialize, initialization_samples)
            tracking_samples = tracker_evaluator.on_initialized(tracker_initialization_results)
            if tracking_samples is not None:
                outputs = _run_fn(model.track, tracking_samples)
            outputs = tracker_evaluator.post_tracking(outputs)

        if data_pipeline_on_host is not None:
            for data_pipeline in reversed(data_pipeline_on_host):
                if hasattr(data_pipeline, 'post_processing'):
                    outputs = data_pipeline.post_processing(outputs)

    def register_data_pipelines(self, branch_name, data_pipelines):
        if 'data_pipeline' in data_pipelines:
            if self.data_pipeline_on_host is None:
                self.data_pipeline_on_host = {}
            if branch_name not in self.data_pipeline_on_host:
                self.data_pipeline_on_host[branch_name] = []
            for data_pipeline in data_pipelines['data_pipeline']:
                self.data_pipeline_on_host[branch_name].append(data_pipeline)
        if 'tracker_evaluator' in data_pipelines:
            if self.tracker_evaluator is None:
                self.tracker_evaluator = {}
            assert branch_name not in self.tracker_evaluator
            self.tracker_evaluator[branch_name] = data_pipelines['tracker_evaluator']
