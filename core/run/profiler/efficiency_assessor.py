import torch
import time
from fvcore.nn import FlopCountAnalysis, flop_count_table


class ModelFlopCountAnalysis:
    def __init__(self, model, pseudo_data_source):
        self.fv_flop_count_analysis_init = FlopCountAnalysis(model, pseudo_data_source.get_init(1))
        self.fv_flop_count_analysis_track = FlopCountAnalysis(model, pseudo_data_source.get_track(1))

    def get_flop_count_table_init(self):
        return flop_count_table(self.fv_flop_count_analysis_init)

    def get_flop_count_table_track(self):
        return flop_count_table(self.fv_flop_count_analysis_track)

    def get_model_mac_init(self):
        return self.fv_flop_count_analysis_init.total()

    def get_model_mac_track(self):
        return self.fv_flop_count_analysis_track.total()


class TrackerEfficiencyAssessor:
    def __init__(self, pseudo_data_source):
        self.pseudo_data_source = pseudo_data_source
        self.batch = 1
        self.flop_count_analysis = None

    def _run_init_loops(self, model, batch, loops):
        init_params, track_params = self.pseudo_data_source.get_eval(batch)
        if self.pseudo_data_source.is_cuda():
            torch.cuda.synchronize()
        begin = time.perf_counter()
        for _ in range(loops):
            with torch.no_grad():
                if isinstance(init_params, (list, tuple)):
                    z_feat = model.initialize(*init_params)
                elif isinstance(init_params, dict):
                    z_feat = model.initialize(**init_params)
                else:
                    z_feat = model.initialize(init_params)
        if self.pseudo_data_source.is_cuda():
            torch.cuda.synchronize()
        end = time.perf_counter()
        return (end - begin) / loops

    def _run_track_loops(self, model, batch, loops):
        init_params, track_params = self.pseudo_data_source.get_eval(batch)
        track_args = []
        track_kwargs = {}
        with torch.no_grad():
            if isinstance(init_params, (list, tuple)):
                z_feat = model.initialize(*init_params)
            elif isinstance(init_params, dict):
                z_feat = model.initialize(**init_params)
            else:
                z_feat = model.initialize(init_params)
        track_args.append(z_feat)
        if isinstance(track_params, (list, tuple)):
            track_args.extend(track_params)
        elif isinstance(track_params, tuple):
            track_kwargs.update(track_params)
        else:
            track_args.append(track_params)
        if self.pseudo_data_source.is_cuda():
            torch.cuda.synchronize()
        begin = time.perf_counter()
        for _ in range(loops):
            with torch.no_grad():
                model.track(*track_args, **track_kwargs)
        if self.pseudo_data_source.is_cuda():
            torch.cuda.synchronize()
        end = time.perf_counter()
        return (end - begin) / loops

    def _test_fps(self, batch, model):
        is_train = model.training
        if is_train:
            model.eval()

        init_time = self._run_init_loops(model, batch, 10)
        init_time = self._run_init_loops(model, batch, 100)
        track_time = self._run_track_loops(model, batch, 10)
        track_time = self._run_track_loops(model, batch, 100)

        if is_train:
            model.train()

        return batch / init_time, batch / track_time

    def get_batch(self):
        return self.batch

    def test_fps(self, model):
        return self._test_fps(1, model)

    def test_fps_batched(self, model):
        return self._test_fps(self.batch, model)

    def get_flop_count_analysis(self, model):
        return ModelFlopCountAnalysis(model, self.pseudo_data_source)
