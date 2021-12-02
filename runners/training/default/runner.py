import math
import torch
from runners.common.branch_utils import get_branch_specific_objects
from core.run.metric_logger.context import get_logger
from runners.interface import BaseRunner


class DefaultTrainer(BaseRunner):
    def __init__(self, criteria, optimizer,
                 lr_scheduler_per_iteration,
                 lr_scheduler_per_epoch, loss_composer,
                 grad_max_norm=None, iteration_step=1):
        self.criteria = criteria
        self.loss_composer = loss_composer

        self.optimizer = optimizer
        self.lr_scheduler_per_iteration = lr_scheduler_per_iteration
        self.lr_scheduler_per_epoch = lr_scheduler_per_epoch

        self.grad_max_norm = grad_max_norm

        self.data_pipeline_on_host = None
        self.branch_name = None
        self.is_train = True
        self.iteration_index = 0
        self.iteration_step = iteration_step

    def get_iteration_index(self):
        return self.iteration_index

    def register_data_pipelines(self, branch_name, data_pipelines):
        if 'data_pipeline' not in data_pipelines:
            return
        if self.data_pipeline_on_host is None:
            self.data_pipeline_on_host = {}
        if branch_name not in self.data_pipeline_on_host:
            self.data_pipeline_on_host[branch_name] = []
        for data_pipeline in data_pipelines['data_pipeline']:
            self.data_pipeline_on_host[branch_name].append(data_pipeline)

    def get_metric_definitions(self):
        if self.is_train:
            runner_metric_definitions = {
                'local': [
                    {'name': 'lr', 'window_size': 1, 'fmt': '{value:.6f}'},
                    {'name': 'loss'}
                ]}
        else:
            runner_metric_definitions = {
                'local': [
                    {'name': 'loss'}
                ]}
        metric_definitions = [runner_metric_definitions]
        data_pipelines = get_branch_specific_objects(self, self.branch_name, 'data_pipeline_on_host')
        if data_pipelines is not None:
            for data_pipeline in data_pipelines:
                if hasattr(data_pipeline, 'get_metric_definitions'):
                    metric_definitions.append(data_pipeline.get_metric_definitions())
        return metric_definitions

    def switch_branch(self, branch_name):
        self.branch_name = branch_name

    def train(self, is_train):
        self.is_train = is_train

    def run_iteration(self, model, data):
        samples, targets, miscellanies_on_host, miscellanies_on_device = data
        data_pipeline_on_host = get_branch_specific_objects(self, self.branch_name, 'data_pipeline_on_host')
        if data_pipeline_on_host is not None:
            for data_pipeline in data_pipeline_on_host:
                if hasattr(data_pipeline, 'pre_processing'):
                    samples, targets, miscellanies_on_host, miscellanies_on_device = data_pipeline.pre_processing(samples, targets, miscellanies_on_host, miscellanies_on_device)

        outputs = None
        loss = None
        if samples is not None:
            if isinstance(samples, (tuple, list)):
                outputs = model(*samples)
            elif isinstance(samples, dict):
                outputs = model(**samples)
            else:
                outputs = model(samples)
            if targets is not None:
                loss, loss_value, loss_stats = self.loss_composer(self.criteria(outputs, targets))
                get_logger().log({'loss': loss_value, **loss_stats})

                if not math.isfinite(loss_value):
                    raise RuntimeError(f"Loss is {loss_value}, stopping training\n{loss_stats}")

        if data_pipeline_on_host is not None:
            for data_pipeline in reversed(data_pipeline_on_host):
                if hasattr(data_pipeline, 'post_processing'):
                    outputs = data_pipeline.post_processing(outputs)

        if loss is not None and self.is_train:
            self.optimizer.zero_grad()
            loss.backward()
            if self.grad_max_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), self.grad_max_norm)
            self.optimizer.step()
            if self.lr_scheduler_per_iteration is not None:
                self.lr_scheduler_per_iteration.step()

            get_logger().log({'lr': self.optimizer.param_groups[0]["lr"]})

        if self.is_train:
            self.iteration_index += self.iteration_step

    def state_dict(self):
        state_dict = {'optimizer': self.optimizer.state_dict(), 'iter': self.iteration_index}
        if self.lr_scheduler_per_iteration is not None:
            state_dict['lr_scheduler_per_iteration'] = self.lr_scheduler_per_iteration.state_dict()
        if self.lr_scheduler_per_epoch is not None:
            state_dict['lr_scheduler'] = self.lr_scheduler_per_epoch.state_dict()
        return state_dict

    def load_state_dict(self, state_dict):
        self.optimizer.load_state_dict(state_dict['optimizer'])
        if self.lr_scheduler_per_iteration is not None:
            self.lr_scheduler_per_iteration.load_state_dict(state_dict['lr_scheduler_per_iteration'])
        if self.lr_scheduler_per_epoch is not None:
            self.lr_scheduler_per_epoch.load_state_dict(state_dict['lr_scheduler'])
        self.iteration_index = state_dict['iter']

    def on_device_changed(self, device):
        self.criteria.to(device)
