from core.run.event_dispatcher.global_event_dispatcher import GlobalEventDispatcher
from core.run.event_dispatcher.register import EventRegister
from core.run.event_dispatcher.runner_event_dispatcher import BranchEventDispatcher
from .profiler.builder import build_profiler, build_efficiency_assessor
import torch
import torch.distributed
from collections import namedtuple
from typing import List, Dict
from core.run.metric_logger.logger import MetricLoggerDispatcher
from runners.interface import BaseRunner
from runners.builder import build_runner
from data.tracking.builder import build_data_source
from models.methods.builder import build_model
from data.tracking.builder import build_data_bridge
from core.run.metric_logger.builder import build_logger
from .running_loop import RunnerDriver


RunningTask = namedtuple('RunningTask', ('data_loader', 'runner', 'logger', 'is_training', 'epoch_interval', 'run_in_last_epoch', 'event_dispatcher'))


def _build_global_event_dispatcher(event_registers):
    started_hooks = []
    finished_hooks = []
    device_changed_hooks = []
    stateful_objects = {}

    for event_register in event_registers:
        if event_register.started_hooks is not None:
            started_hooks.extend(event_register.started_hooks)
        if event_register.finished_hooks is not None:
            finished_hooks.extend(event_register.finished_hooks)
        if event_register.device_changed_hooks is not None:
            device_changed_hooks.extend(event_register.device_changed_hooks)
        if event_register.stateful_objects is not None:
            stateful_objects.update(event_register.stateful_objects)
    if len(started_hooks) == 0:
        started_hooks = None
    if len(finished_hooks) == 0:
        finished_hooks = None
    if len(device_changed_hooks) == 0:
        device_changed_hooks = None
    if len(stateful_objects) == 0:
        stateful_objects = None
    return GlobalEventDispatcher(started_hooks, finished_hooks, stateful_objects, device_changed_hooks)


def _build_branch_event_dispatcher(event_registers):
    epoch_begin_hooks = []
    epoch_end_hooks = []
    iteration_begin_hooks = []
    iteration_end_hooks = []
    status_collectors = {}

    for event_register in event_registers:
        if event_register.epoch_begin_hooks is not None:
            epoch_begin_hooks.extend(event_register.epoch_begin_hooks)
        if event_register.epoch_end_hooks is not None:
            epoch_end_hooks.extend(event_register.epoch_end_hooks)
        if event_register.iteration_begin_hooks is not None:
            iteration_begin_hooks.extend(event_register.iteration_begin_hooks)
        if event_register.iteration_end_hooks is not None:
            iteration_end_hooks.extend(event_register.iteration_end_hooks)
        if event_register.status_collectors is not None:
            status_collectors.update(event_register.status_collectors)
    if len(epoch_begin_hooks) is None:
        epoch_begin_hooks = None
    if len(epoch_end_hooks) is None:
        epoch_end_hooks = None
    if len(iteration_begin_hooks) is None:
        iteration_begin_hooks = None
    if len(iteration_end_hooks) is None:
        iteration_end_hooks = None
    if len(status_collectors) is None:
        status_collectors = None
    return BranchEventDispatcher(epoch_begin_hooks, epoch_end_hooks, iteration_begin_hooks, iteration_end_hooks, status_collectors)


def _check_running_branches(branch_config):
    has_training_run = False

    training_runner = set()
    reordered_branch_config = {}  # rely on insertion order behaviour
    for branch_name, sub_branch_config in branch_config.items():
        is_training = sub_branch_config['training']
        runner_name = sub_branch_config['runner']
        if is_training:
            has_training_run = True
            assert runner_name not in training_runner, "one runner can only be associated with one training branch"
            training_runner.add(runner_name)
            reordered_branch_config[branch_name] = sub_branch_config

    for branch_name, sub_branch_config in branch_config.items():
        is_training = sub_branch_config['training']
        if not is_training:
            reordered_branch_config[branch_name] = sub_branch_config

    return reordered_branch_config, has_training_run


class DataContext:
    name: str
    context: dict
    event_register: EventRegister
    data_loader: object
    host_data_pipelines: dict

    def __init__(self, name):
        self.name = name
        self.context = {}
        self.event_register = EventRegister(f'data/{name}/')


class BranchContext:
    event_register: EventRegister
    data_source_name: str
    runner_name: str
    is_training: bool
    logger: MetricLoggerDispatcher
    epoch_interval: int
    data_bridge: List[object]

    def __init__(self, name, is_training, data_source_name, runner_name, epoch_interval):
        self.event_register = EventRegister(f'branch/{name}/')
        self.is_training = is_training
        self.data_source_name = data_source_name
        self.runner_name = runner_name
        self.epoch_interval = epoch_interval


class RunnerContext:
    name: str
    event_register: EventRegister
    runner: BaseRunner

    def __init__(self, name):
        self.name = name
        self.event_register = EventRegister(f'runner/{name}/')


class BuildingContext:
    branch_contexts: Dict[str, BranchContext]
    data_source_contexts: Dict[str, DataContext]
    runner_contexts: Dict[str, RunnerContext]
    model_event_register: EventRegister
    logger_event_register: EventRegister
    model: object
    has_training_run: bool
    num_epochs: int
    pseudo_data_generator: object

    def __init__(self, num_epochs):
        self.data_source_contexts = {}
        self.branch_contexts = {}
        self.runner_contexts = {}
        self.logger_event_register = EventRegister('logger/')
        self.model_event_register = EventRegister('model/')
        self.num_epochs = num_epochs


def _pre_build_branch_context(branch_config, building_context: BuildingContext):
    for branch_name, sub_branch_config in branch_config.items():
        is_training = sub_branch_config['training']
        data_loader_name = sub_branch_config['data']
        runner_name = sub_branch_config['runner']
        epoch_interval = 1
        if 'epoch_interval' in sub_branch_config:
            epoch_interval = sub_branch_config['epoch_interval']

        building_context.branch_contexts[branch_name] = BranchContext(branch_name, is_training, data_loader_name, runner_name, epoch_interval)


def _build_data_loaders(runtime_vars, branch_config: dict, data_config: dict, config: dict, global_rng, local_rng, building_context: BuildingContext):
    for branch_name, sub_branch_config in branch_config.items():
        is_training = sub_branch_config['training']
        data_loader_name = sub_branch_config['data']

        if data_loader_name not in building_context.data_source_contexts:
            data_context = DataContext(data_loader_name)
            data_loader, host_data_pipelines = build_data_source(  # host_data_pipelines = [ { type: instance } ]
                data_config[data_loader_name], runtime_vars, config, global_rng, local_rng,
                data_context.event_register, data_context.context, is_training)
            data_context.data_loader = data_loader
            data_context.host_data_pipelines = host_data_pipelines
            building_context.data_source_contexts[data_loader_name] = data_context


def _build_model(config, runtime_vars, building_context: BuildingContext):
    has_training_run = False
    num_epochs = building_context.num_epochs
    iterations_per_epoch = None
    batch_size = None
    for branch_context in building_context.branch_contexts.values():
        if branch_context.is_training:
            data_source_context = building_context.data_source_contexts[branch_context.data_source_name]
            iterations_per_epoch = data_source_context.context['iterations_per_epoch']
            batch_size = data_source_context.context['batch_size']
            break

    model, pseudo_data_generator = build_model(config, runtime_vars, batch_size, num_epochs, iterations_per_epoch, building_context.model_event_register, has_training_run)
    
    weight_path = runtime_vars.weight_path
    if weight_path is not None:
        model.load_state_dict(torch.load(weight_path, map_location='cpu')['model'])

    device = torch.device(runtime_vars.device)
    model.to(device)

    building_context.model = model
    building_context.pseudo_data_generator = pseudo_data_generator


def _build_runners(runtime_vars, branch_config: dict, data_config: dict, runner_config: dict, config: dict,
                   building_context: BuildingContext, wandb_instance):
    model = building_context.model
    for branch_name, sub_branch_config in branch_config.items():
        branch_context = building_context.branch_contexts[branch_name]
        data_source_name = branch_context.data_source_name
        data_source_context = building_context.data_source_contexts[data_source_name]
        runner_name = branch_context.runner_name
        if runner_name not in building_context.runner_contexts:
            runner_context = RunnerContext(runner_name)

            runner = build_runner(model, runner_config[runner_name], data_source_context.context, config, runner_context.event_register)
            runner_context.runner = runner
            building_context.runner_contexts[runner_name] = runner_context
        else:
            runner_context = building_context.runner_contexts[runner_name]

        data_bridges = build_data_bridge(runtime_vars, sub_branch_config, data_config[data_source_name], runner_config[runner_name], config, branch_context.event_register, data_source_context.context, building_context.has_training_run)

        runner = runner_context.runner
        host_data_pipelines = data_source_context.host_data_pipelines
        if host_data_pipelines is not None:
            runner.register_data_pipelines(branch_name, host_data_pipelines)
        if data_bridges is not None:
            runner.register_data_pipelines(branch_name, data_bridges)

        logging_config = None
        if 'logging' in sub_branch_config:
            logging_config = sub_branch_config['logging']
        logger = build_logger(logging_config, wandb_instance, branch_context.event_register, building_context.logger_event_register)

        runner.switch_branch(branch_name)
        runner.train(branch_context.is_training)
        runner_metric_definitions = runner.get_metric_definitions()
        if runner_metric_definitions is not None and len(runner_metric_definitions) > 0:
            for runner_metric_definition in runner_metric_definitions:
                logger.register_metric(runner_metric_definition)
        branch_context.logger = logger


def build_running_tasks(runtime_vars, config: dict, global_rng, local_rng, wandb_instance):
    building_context = BuildingContext(config['runs']['num_epochs'])
    branch_config = config['runs']['branch']
    data_config = config['runs']['data']
    runner_config = config['runs']['runner']

    branch_config, has_training_run = _check_running_branches(branch_config)
    building_context.has_training_run = has_training_run
    _pre_build_branch_context(branch_config, building_context)
    _build_data_loaders(runtime_vars, branch_config, data_config, config, global_rng, local_rng, building_context)
    _build_model(config, runtime_vars, building_context)
    _build_runners(runtime_vars, branch_config, data_config, runner_config, config, building_context, wandb_instance)

    default_logger = build_logger(None, wandb_instance, building_context.logger_event_register, building_context.logger_event_register)

    all_event_register = [building_context.model_event_register]
    for branch_context in building_context.branch_contexts.values():
        all_event_register.append(branch_context.event_register)
    for data_context in building_context.data_source_contexts.values():
        all_event_register.append(data_context.event_register)
    for runner_context in building_context.runner_contexts.values():
        all_event_register.append(runner_context.event_register)
    all_event_register.append(building_context.logger_event_register)
    global_event_dispatcher = _build_global_event_dispatcher(all_event_register)

    running_tasks = {}
    for branch_name, branch_context in building_context.branch_contexts.items():
        epoch_interval = branch_context.epoch_interval
        run_in_last_epoch = False
        if epoch_interval < 0:
            run_in_last_epoch = True
            epoch_interval = 0

        data_source_context = building_context.data_source_contexts[branch_context.data_source_name]
        runner_context = building_context.runner_contexts[branch_context.runner_name]

        branch_event_registers = [branch_context.event_register, data_source_context.event_register, runner_context.event_register, building_context.model_event_register, building_context.logger_event_register]
        branch_event_dispatcher = _build_branch_event_dispatcher(branch_event_registers)
        running_tasks[branch_name] = RunningTask(data_source_context.data_loader, runner_context.runner, branch_context.logger, branch_context.is_training, epoch_interval, run_in_last_epoch, branch_event_dispatcher)

    model = building_context.model
    pseudo_data_generator = building_context.pseudo_data_generator

    if 'sync_bn' in config and 'cuda' in runtime_vars.device:
        if config['sync_bn']:
            from miscellanies.torch.distributed import is_dist_available_and_initialized
            if is_dist_available_and_initialized():
                model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    if runtime_vars.distributed:
        if 'cuda' in runtime_vars.device:
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[runtime_vars.local_rank])
        else:
            model = torch.nn.parallel.DistributedDataParallel(model)

    return model, pseudo_data_generator, running_tasks, global_event_dispatcher, default_logger, building_context.num_epochs


def build(runtime_vars, config: dict, global_rng, local_rng, wandb_instance):
    model, pseudo_data_generator, running_tasks, global_event_dispatcher, default_logger, num_epochs = build_running_tasks(runtime_vars, config, global_rng, local_rng, wandb_instance)
    profiler = build_profiler(runtime_vars)

    efficiency_assessor = build_efficiency_assessor(pseudo_data_generator)

    if runtime_vars.enable_autograd_detect_anomaly:
        print('The anomaly detection for the autograd engine enabled.')
        torch.autograd.set_detect_anomaly(True)

    return RunnerDriver(config['name'], num_epochs, model, running_tasks, global_event_dispatcher, runtime_vars, efficiency_assessor, wandb_instance, profiler, default_logger)
