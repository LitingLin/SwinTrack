import torch


class ResponseMapTrackingPostProcessing:
    def __init__(self, enable_gaussian_score_map_penalty, search_feat_size, window_penalty_ratio=None):
        self.enable_gaussian_score_map_penalty = enable_gaussian_score_map_penalty
        self.search_feat_size = search_feat_size

        if enable_gaussian_score_map_penalty:
            self.window = torch.flatten(torch.outer(torch.hann_window(search_feat_size[1], periodic=False),
                                                    torch.hann_window(search_feat_size[0], periodic=False)))

            self.window_penalty_ratio = window_penalty_ratio

    def __call__(self, network_output):
        if network_output is None:
            return None
        class_score_map, predicted_bbox = network_output['class_score'], network_output['bbox']  # shape: (N, 1, H, W), (N, H, W, 4)
        N, C, H, W = class_score_map.shape
        assert C == 1
        class_score_map = class_score_map.view(N, H * W)

        if self.enable_gaussian_score_map_penalty:
            # window penalty
            class_score_map = class_score_map * (1 - self.window_penalty_ratio) + \
                     self.window.view(1, H * W) * self.window_penalty_ratio

        confidence_score, best_idx = torch.max(class_score_map, 1)

        predicted_bbox = predicted_bbox.view(N, H * W, 4)
        bounding_box = predicted_bbox[torch.arange(len(predicted_bbox)), best_idx, :]
        processor_outputs = {'bbox': bounding_box, 'conf': confidence_score}
        for k, v in network_output.items():
            if k not in ('class_score', 'bbox'):
                processor_outputs[k] = v
        return processor_outputs

    def to(self, device):
        if self.enable_gaussian_score_map_penalty:
            self.window = self.window.to(device)
