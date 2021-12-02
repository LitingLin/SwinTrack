from queue import Queue
import threading
import numpy as np
from collections import namedtuple
from miscellanies.torch.distributed import get_world_size
import torch.distributed
from core.run.metric_logger.context import get_logger


SequenceEvaluationMetrics = namedtuple('SequenceEvaluationMetrics',
                                       ('average_overlap',
                                        'success_rate_at_iou_0_5', 'success_rate_at_iou_0_75',
                                        'success_curve', 'precision_curve', 'normalized_precision_curve', 'fps'))


class EvaluationMetricsCollector:
    def __init__(self, tracker_name, dataset_sequence_names, on_the_fly_constructor, multiple_run=False, saving_path: str = None, summary_by_mean=False):
        self.tracker_name = tracker_name
        self.multiple_run = multiple_run
        self.saving_path = saving_path
        self.world_size = get_world_size()
        self.gather_interval = 32
        self.summary_by_mean = summary_by_mean
        self.dataset_names = dataset_sequence_names
        self.on_the_fly_constructor = on_the_fly_constructor
        self.sub_collectors = None
        self.sequence_distributor = None

    def on_started(self):
        self.current_epoch = None
        self.runs = {}

    def on_epoch_begin(self, epoch):
        assert self.current_epoch is None
        self.current_epoch = epoch

        self.iteration_index = 0

        self.thread = None
        self.task_queue = None
        if self.saving_path is not None:
            self.task_queue = Queue(16)
            self.thread = threading.Thread(target=self._worker_entry)
            self.thread.start()
        constructor_epoch_param = None
        if self.multiple_run:
            constructor_epoch_param = epoch
        self.sequence_distributor, self.sub_collectors = self.on_the_fly_constructor.construct(self.dataset_names, self.saving_path, self.tracker_name, constructor_epoch_param)

    def on_epoch_end(self, epoch):
        assert self.current_epoch == epoch

        if self.iteration_index % self.gather_interval != 0:
            self._do_gathering()

        if self.saving_path is not None:
            self.task_queue.put('quit')
            self.thread.join()
            del self.thread
            del self.task_queue

        summary_metrics = {}
        for sub_collector in self.sub_collectors:
            sub_summary = sub_collector.get_summary()
            if sub_summary is not None:
                summary_metrics.update(sub_summary)
        if len(summary_metrics) > 0:
            if self.multiple_run:
                get_logger().log(wandb={'epoch': epoch, **summary_metrics})
            self.runs[epoch] = summary_metrics
        self.sub_collectors = None
        self.sequence_distributor = None
        self.current_epoch = None

    def on_finished(self):
        summary = self._get_summary()
        if summary is not None:
            get_logger().log_summary(summary)
        self.runs = None

    def _get_summary(self):
        if len(self.runs) == 0:
            return None
        # do with summary
        if self.summary_by_mean:
            summary = {}
            keys = (next(iter(self.runs.values()))).keys()
            for key in keys:
                summary[key] = np.mean([run[key] for run in self.runs.values()])
            return summary
        else:
            return self.runs[max(self.runs.keys())]

    def post_processing(self, collected_sequences):
        self.iteration_index += 1
        if collected_sequences is not None and len(collected_sequences) != 0:
            self.sequence_distributor(collected_sequences, self.sub_collectors, self.task_queue)

        if self.iteration_index % self.gather_interval == 0:
            self._do_gathering()

        return collected_sequences

    def _do_gathering(self):
        gathering_objects = [sub_collector.prepare_gathering() for sub_collector in self.sub_collectors]

        if self.world_size > 1:
            gathered_objects = [None] * self.world_size
            torch.distributed.all_gather_object(gathered_objects, gathering_objects)
            collated = []
            for index in range(len(self.sub_collectors)):
                sub_collated = []
                for gathered_object in gathered_objects:
                    sub_collector_objects = gathered_object[index]
                    if sub_collector_objects is not None:
                        sub_collated.extend(sub_collector_objects)
                if len(sub_collated) == 0:
                    sub_collated = None
                collated.append(sub_collated)
            gathered_objects = collated
        else:
            gathered_objects = gathering_objects

        logging_metrics = {}
        for gathered_object, sub_collector in zip(gathered_objects, self.sub_collectors):
            if gathered_object is not None:
                logging_metric = sub_collector.accept_gathered(gathered_object, self.task_queue)
                if logging_metric is not None:
                    logging_metrics.update(logging_metric)

        if len(logging_metrics) > 0:
            logger = get_logger()
            local_metric_definitions = []
            for logging_metric_name in logging_metrics.keys():
                local_metric_definitions.append({'name': logging_metric_name, 'window_size': 1, 'fmt': '{value:.4f}'})
            logger.register_metric({'local': local_metric_definitions})
            get_logger().log(local=logging_metrics)

    def _worker_entry(self):
        while True:
            work = self.task_queue.get()
            if work == 'quit':
                return
            if len(work) == 1:
                work[0]()
            else:
                work[0](*work[1:])
            self.task_queue.task_done()
