from miscellanies.torch.distributed import is_main_process
from .utils.calculate_metrics import calculate_evaluation_metrics
from .utils.draw_plots import draw_success_plot, draw_precision_plot, draw_normalized_precision_plot
import os
import csv
from collections import namedtuple
import numpy as np
import json
from typing import List


SequenceEvaluationMetrics = namedtuple('SequenceEvaluationMetrics',
                                       ('average_overlap',
                                        'success_rate_at_iou_0_5', 'success_rate_at_iou_0_75',
                                        'success_curve', 'precision_curve', 'normalized_precision_curve', 'fps'))


class FullyAnnotatedDatasetReportGenerator:
    def __init__(self, datasets: dict, tracker_name, saving_path: str = None):
        self.datasets = datasets
        self.saving_path = saving_path
        self.tracker_name = tracker_name
        self.datasets_evaluation_report_generated = False
        self.cached_sequence_metrics = []
        self.run_results = {}
        self.sequence_summaries = {}

    def accept_evaluated_sequence(self, collected_sequences, io_thread):
        if io_thread is not None and self.saving_path is not None and is_main_process():
            if not self.datasets_evaluation_report_generated:
                io_thread.put((self._generate_datasets_evaluation_report_header, ))
                self.datasets_evaluation_report_generated = True

        if collected_sequences is not None and len(collected_sequences) > 0:
            for dataset_unique_id, sequence_name, object_existence, groundtruth_bboxes, predicted_bboxes, time_cost_array in collected_sequences:
                object_existence = object_existence.numpy()
                groundtruth_bboxes = groundtruth_bboxes.numpy()
                predicted_bboxes = predicted_bboxes.numpy()
                time_cost_array = time_cost_array.numpy()

                ao, sr_at_0_5, sr_at_0_75, succ_curve, prec_curve, norm_prec_curve = calculate_evaluation_metrics(predicted_bboxes, groundtruth_bboxes, object_existence)

                if io_thread is not None and self.saving_path is not None:
                    io_thread.put(
                        (self._save_sequence_predicted_bboxes, dataset_unique_id, sequence_name, predicted_bboxes, ao, sr_at_0_5, sr_at_0_75, succ_curve, prec_curve, norm_prec_curve, time_cost_array))

                self.cached_sequence_metrics.append((dataset_unique_id, sequence_name, SequenceEvaluationMetrics(ao, sr_at_0_5, sr_at_0_75, succ_curve, prec_curve, norm_prec_curve,  1.0 / np.mean(time_cost_array))))

    def _generate_datasets_evaluation_report_header(self):
        os.makedirs(self.saving_path, exist_ok=True)
        with open(os.path.join(self.saving_path, 'performance.csv'), 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(('Dataset Name', 'Success Score', 'Precision Score', 'Normalized Precision Score',
                             'Average Overlap', 'Success Rate @ IOU>=0.5', 'Success Rate @ IOU>=0.75', 'FPS'))

    def prepare_gathering(self):
        return self.cached_sequence_metrics

    def accept_gathered(self, gathered_sequence_metrics, io_thread):
        logging_metrics = {}
        for dataset_unique_id, sequence_name, sequence_evaluation_metrics in gathered_sequence_metrics:
            print(f'{sequence_name}: success {np.mean(sequence_evaluation_metrics.success_curve):.4f} prec {sequence_evaluation_metrics.precision_curve[20]:.4f} norm prec {sequence_evaluation_metrics.normalized_precision_curve[20]:.4f} ')
            if dataset_unique_id not in self.run_results:
                self.run_results[dataset_unique_id] = ([], [])
            self.run_results[dataset_unique_id][0].append(sequence_name)
            self.run_results[dataset_unique_id][1].append(sequence_evaluation_metrics)

            if len(self.run_results[dataset_unique_id][0]) == len(self.datasets[dataset_unique_id]):
                success_score, precision_score, normalized_precision_score, average_overlap, success_rate_at_iou_0_5, success_rate_at_iou_0_75, fps = \
                    self._generate_dataset_evaluation_metrics(dataset_unique_id, io_thread, *self.run_results[dataset_unique_id])
                logging_metrics[f'success_score_{dataset_unique_id}'] = success_score
                logging_metrics[f'precision_score_{dataset_unique_id}'] = precision_score
                logging_metrics[f'norm_precision_score_{dataset_unique_id}'] = normalized_precision_score
                logging_metrics[f'average_overlap_{dataset_unique_id}'] = average_overlap
                logging_metrics[f'success_rate_at_iou_0_5_{dataset_unique_id}'] = success_rate_at_iou_0_5
                logging_metrics[f'success_rate_at_iou_0_75_{dataset_unique_id}'] = success_rate_at_iou_0_75
                logging_metrics[f'fps_{dataset_unique_id}'] = fps
            assert len(self.run_results[dataset_unique_id][0]) <= len(self.datasets[dataset_unique_id])
        if len(logging_metrics) == 0:
            logging_metrics = None
        else:
            self.sequence_summaries.update(logging_metrics)

        self.cached_sequence_metrics.clear()  # self.cached_sequence_metrics & gathered_sequence_metrics may be the same object
        return logging_metrics

    def get_summary(self):
        sequence_evaluation_metrics = []
        [sequence_evaluation_metrics.extend(sequence[1]) for sequence in self.run_results.values()]

        _, _, _, success_score, precision_score, normalized_precision_score, average_overlap, success_rate_at_iou_0_5, success_rate_at_iou_0_75, fps = \
            self._calculate_metrics(sequence_evaluation_metrics)
        summary = {'success_score': success_score, 'precision_score': precision_score,
                   'norm_precision_score': normalized_precision_score,
                   'average_overlap': average_overlap,
                   'success_rate_at_iou_0_5': success_rate_at_iou_0_5,
                   'success_rate_at_iou_0_75': success_rate_at_iou_0_75}
        summary.update(self.sequence_summaries)
        return summary

    def _save_sequence_predicted_bboxes(self, dataset_unique_id, sequence_name, predicted_bboxes,
                                        ao, sr_at_0_5, sr_at_0_75, succ_curve, prec_curve, norm_prec_curve,
                                        time_cost_array):
        saving_path = os.path.join(self.saving_path, dataset_unique_id, sequence_name)
        os.makedirs(saving_path, exist_ok=True)

        np.save(os.path.join(saving_path, 'bounding_box.npy'), predicted_bboxes)
        np.savetxt(os.path.join(saving_path, 'bounding_box.txt'), predicted_bboxes, fmt='%.3f', delimiter=',')

        np.save(os.path.join(saving_path, 'time.npy'), time_cost_array)
        np.savetxt(os.path.join(saving_path, 'time.txt'), time_cost_array, fmt='%.3f', delimiter=',')

        assert len(predicted_bboxes) == len(time_cost_array)

        with open(os.path.join(saving_path, 'performance.json'), 'w', encoding='utf-8', newline='') as f:
            sequence_report = {
                'average_overlap': ao,
                'success_rate_at_iou_0.5': sr_at_0_5,
                'success_rate_at_iou_0.75': sr_at_0_75,
                'success_curve': succ_curve.tolist(),
                'precision_curve': prec_curve.tolist(),
                'normalized_precision_curve': norm_prec_curve.tolist(),
                'success_score': np.mean(succ_curve),
                'precision_score': prec_curve[20],  # center location error @ 20 pix
                'normalized_precision_score': norm_prec_curve[20],
                'fps': 1.0 / np.mean(time_cost_array)
            }
            json.dump(sequence_report, f, indent=2)

    def _generate_dataset_evaluation_report(self, dataset_unique_id, sequence_names: List[str], sequence_evaluation_metrics: List[SequenceEvaluationMetrics],
                                            success_curve, precision_curve, normalized_precision_curve,
                                            success_score, precision_score, normalized_precision_score,
                                            average_overlap, success_rate_at_iou_0_5, success_rate_at_iou_0_75, fps):
        dataset_report_path = os.path.join(self.saving_path, dataset_unique_id)
        with open(os.path.join(dataset_report_path, 'sequences_performance.csv'), 'w', newline='', encoding='utf-8') as f:
            csv_writer = csv.writer(f)
            csv_writer.writerow(('Sequence Name', 'Success Score', 'Precision Score', 'Normalized Precision Score',
                                 'Average Overlap', 'Success Rate @ IOU>=0.5', 'Success Rate @ IOU>=0.75', 'FPS'))
            for sequence_name, sequence_evaluation_metric in zip(sequence_names, sequence_evaluation_metrics):
                csv_writer.writerow((sequence_name, np.mean(sequence_evaluation_metric.success_curve),
                                     sequence_evaluation_metric.precision_curve[20],
                                     sequence_evaluation_metric.normalized_precision_curve[20],
                                     sequence_evaluation_metric.average_overlap,
                                     sequence_evaluation_metric.success_rate_at_iou_0_5,
                                     sequence_evaluation_metric.success_rate_at_iou_0_75,
                                     sequence_evaluation_metric.fps))

        draw_success_plot(success_curve[np.newaxis, :], [self.tracker_name], dataset_report_path)
        draw_precision_plot(precision_curve[np.newaxis, :], [self.tracker_name], dataset_report_path)
        draw_normalized_precision_plot(normalized_precision_curve[np.newaxis, :], [self.tracker_name], dataset_report_path)

        dataset_report = {'success_score': success_score,
                          'precision_score': precision_score,
                          'normalized_precision_score': normalized_precision_score,
                          'average_overlap': average_overlap,
                          'success_rate_at_iou_0.5': success_rate_at_iou_0_5,
                          'success_rate_at_iou_0.75': success_rate_at_iou_0_75,
                          'success_curve': success_curve.tolist(),
                          'precision_curve': precision_curve.tolist(),
                          'normalized_precision_curve': normalized_precision_curve.tolist(),
                          'fps': fps}
        with open(os.path.join(dataset_report_path, 'performance.json'), 'w', newline='', encoding='utf-8') as f:
            json.dump(dataset_report, f, indent=2)

        with open(os.path.join(self.saving_path, 'performance.csv'), 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow((dataset_unique_id, success_score, precision_score, normalized_precision_score, average_overlap, success_rate_at_iou_0_5, success_rate_at_iou_0_75, fps))

    def _generate_dataset_evaluation_metrics(self, dataset_unique_id, io_thread, sequence_names: List[str], sequence_evaluation_metrics: List[SequenceEvaluationMetrics]):
        success_curve, precision_curve, normalized_precision_curve, success_score, precision_score, normalized_precision_score, average_overlap, success_rate_at_iou_0_5, success_rate_at_iou_0_75, fps = \
            self._calculate_metrics(sequence_evaluation_metrics)
        if is_main_process():
            if self.saving_path is not None and io_thread is not None:
                io_thread.put((self._generate_dataset_evaluation_report, dataset_unique_id, sequence_names, sequence_evaluation_metrics,
                                     success_curve, precision_curve, normalized_precision_curve,
                                     success_score, precision_score, normalized_precision_score,
                                     average_overlap, success_rate_at_iou_0_5, success_rate_at_iou_0_75, fps))
        return success_score, precision_score, normalized_precision_score, average_overlap, success_rate_at_iou_0_5, success_rate_at_iou_0_75, fps

    @staticmethod
    def _calculate_metrics(sequence_evaluation_metrics: List[SequenceEvaluationMetrics]):
        success_curve = np.mean([sequence_evaluation_metric.success_curve for sequence_evaluation_metric in sequence_evaluation_metrics], axis=0)
        precision_curve = np.mean([sequence_evaluation_metric.precision_curve for sequence_evaluation_metric in sequence_evaluation_metrics], axis=0)
        normalized_precision_curve = np.mean([sequence_evaluation_metric.normalized_precision_curve for sequence_evaluation_metric in sequence_evaluation_metrics], axis=0)

        success_score = np.mean(success_curve)
        precision_score = precision_curve[20]
        normalized_precision_score = normalized_precision_curve[20]

        average_overlap = np.mean([sequence_evaluation_metric.average_overlap for sequence_evaluation_metric in sequence_evaluation_metrics])
        success_rate_at_iou_0_5 = np.mean([sequence_evaluation_metric.success_rate_at_iou_0_5 for sequence_evaluation_metric in sequence_evaluation_metrics])
        success_rate_at_iou_0_75 = np.mean([sequence_evaluation_metric.success_rate_at_iou_0_75 for sequence_evaluation_metric in sequence_evaluation_metrics])
        fps = np.mean([sequence_evaluation_metric.fps for sequence_evaluation_metric in sequence_evaluation_metrics])

        return success_curve, precision_curve, normalized_precision_curve, success_score, precision_score, normalized_precision_score, average_overlap, success_rate_at_iou_0_5, success_rate_at_iou_0_75, fps
