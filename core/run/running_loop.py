import sys
import time
import torch.distributed
from miscellanies.torch.checkpoint import dump_checkpoint, load_checkpoint
from tqdm import tqdm
import datetime
from contextlib import nullcontext
from core.run.metric_logger.context import enable_logger
from torch import set_grad_enabled
import gc


def print_model_efficiency_assessment(efficiency_assessor, model, wandb_instance):
    flop_count_analysis = efficiency_assessor.get_flop_count_analysis(model)
    print('Initialization modules flop table')
    print(flop_count_analysis.get_flop_count_table_init())

    print('Tracking modules flop table')
    print(flop_count_analysis.get_flop_count_table_track())

    init_fps, track_fps = efficiency_assessor.test_fps(model)

    if wandb_instance is not None:
        wandb_instance.summary.update({'model_mac_init': flop_count_analysis.get_model_mac_init(), 'model_mac_track': flop_count_analysis.get_model_mac_track()})

    print(f"Estimated model FPS: init {init_fps:.3f} track {track_fps:.3f}")


def _wandb_watch_model(model, wandb_instance, watch_model_parameters, watch_model_gradients, watch_model_freq):
    if watch_model_parameters and watch_model_gradients:
        watch_model = 'all'
    elif watch_model_parameters:
        watch_model = 'parameters'
    elif watch_model_gradients:
        watch_model = 'gradients'
    else:
        watch_model = None

    wandb_instance.watch(model, log=watch_model, log_freq=watch_model_freq)


def get_model(model: torch.nn.Module):
    if model.__class__.__name__ == 'DistributedDataParallel':
        return model.module
    else:
        return model


def run_iteration(model, data_loader, runner, branch_name, event_dispatcher, logger, is_training, epoch):
    with enable_logger(logger), set_grad_enabled(is_training):
        runner.switch_branch(branch_name)
        runner.train(is_training)
        model.train(is_training)
        if not is_training:
            model = get_model(model)
        event_dispatcher.epoch_begin(epoch)
        for data in logger.loggers['local'].log_every(data_loader):
            event_dispatcher.iteration_begin(is_training)
            logger.set_step(runner.get_iteration_index())
            runner.run_iteration(model, data)
            event_dispatcher.iteration_end(is_training)
        event_dispatcher.epoch_end(epoch)
        gc.collect()
        epoch_status = event_dispatcher.collect_status()
        if epoch_status is not None and len(epoch_status) > 0:
            print(f'Epoch {epoch} branch {branch_name} statistics:')
            for status_name, status in epoch_status.items():
                print('----------------------------')
                print(f'{status_name}:')
                print(status)
                print('----------------------------')


class RunnerDriver:
    def __init__(self, name, n_epochs, model, runs, event_dispatcher, runtime_vars, efficiency_assessor, wandb_instance,
                 profiler, default_logger):
        self.name = name
        self.model = model
        self.event_dispatcher = event_dispatcher
        self.runtime_vars = runtime_vars
        self.n_epochs = n_epochs
        self.runs = runs
        self.wandb_instance = wandb_instance
        self.efficiency_assessor = efficiency_assessor
        self.epoch = 0
        if profiler is None:
            self.profiler = nullcontext()
        else:
            self.profiler = profiler
        self.dumping_interval = runtime_vars.checkpoint_interval
        self.default_logger = default_logger
        self.output_path = runtime_vars.output_dir
        self.resume_path = runtime_vars.resume
        self.device = torch.device(runtime_vars.device)

    def run(self):
        if self.resume_path is not None:
            model_state_dict, objects_state_dict = load_checkpoint(self.resume_path)
            assert model_state_dict['version'] == 2
            get_model(self.model).load_state_dict(model_state_dict['model'])
            self.epoch = objects_state_dict['epoch']
            self.event_dispatcher.dispatch_state_dict(objects_state_dict)

        self.event_dispatcher.device_changed(self.device)

        has_training_run = False
        for run in self.runs.values():
            is_training_run = run.is_training
            if is_training_run:
                has_training_run = True

        if has_training_run:
            print("Start training")
        else:
            print("Start evaluation")
        if self.wandb_instance is not None:
            _wandb_watch_model(get_model(self.model), self.wandb_instance, self.runtime_vars.watch_model_parameters,
                               self.runtime_vars.watch_model_gradients, self.runtime_vars.watch_model_freq)

        print_model_efficiency_assessment(self.efficiency_assessor, get_model(self.model), self.wandb_instance)

        start_time = time.perf_counter()

        if has_training_run:
            description = f'Train {self.name}'
        else:
            description = f'Evaluate {self.name}'
        with enable_logger(self.default_logger), self.event_dispatcher, self.profiler:
            for epoch in tqdm(range(self.epoch, self.n_epochs), desc=description, file=sys.stdout):
                print()
                self.epoch = epoch

                epoch_has_training_run = False
                for branch_name, (data_loader, runner, logger, is_training, epoch_interval, run_in_last_epoch, event_dispatcher) in self.runs.items():
                    assert epoch_interval >= 0
                    if is_training:
                        epoch_has_training_run = True
                    if (run_in_last_epoch and epoch + 1 == self.n_epochs) or (epoch_interval != 0 and epoch % epoch_interval == 0):
                        run_iteration(self.model, data_loader, runner, branch_name, event_dispatcher, logger, is_training, epoch)

                if epoch_has_training_run and self.output_path is not None and (epoch % self.dumping_interval == 0 or epoch + 1 == self.n_epochs):
                    model_state_dict = {'version': 2, 'model': get_model(self.model).state_dict()}
                    objects_state_dict = {'epoch': epoch}
                    additional_objects_state_dict = self.event_dispatcher.collect_state_dict()

                    if additional_objects_state_dict is not None:
                        objects_state_dict.update(additional_objects_state_dict)
                    dump_checkpoint(model_state_dict, objects_state_dict, epoch, self.output_path)

        total_time = time.perf_counter() - start_time
        total_time_str = str(datetime.timedelta(seconds=total_time))
        if has_training_run:
            print(f'Training time {total_time_str}')
        else:
            print(f'Evaluating time {total_time_str}')
