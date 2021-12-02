import torch


class SiamFCPseudoDataGenerator:
    def __init__(self, template_size, search_size, template_feat_map_size, hidden_dim):
        self.template_size = template_size
        self.search_size = search_size
        self.template_feat_map_size = template_feat_map_size
        self.device = torch.device('cpu')
        self.dim = hidden_dim

    def get_train(self, batch):
        return torch.full((batch, 3, *self.template_size), 0.5, device=self.device), \
            torch.full((batch, 3, *self.search_size), 0.5, device=self.device)

    def get_eval(self, batch):
        return torch.full((batch, 3, *self.template_size), 0.5, device=self.device), \
            torch.full((batch, 3, *self.search_size), 0.5, device=self.device)

    def get_init(self, batch):
        return torch.full((batch, 3, *self.template_size), 0.5, device=self.device), None, None

    def get_track(self, batch):
        return None, torch.full((batch, 3, *self.search_size), 0.5, device=self.device), torch.full((batch, self.template_feat_map_size[0] * self.template_feat_map_size[1], self.dim), 0.5, device=self.device)

    def is_cuda(self):
        return 'cuda' in self.device.type

    def get_device(self):
        return self.device

    def on_device_changed(self, device):
        self.device = device


def build_siamfc_pseudo_data_generator(network_config: dict, event_register):
    network_data_config = network_config['data']
    transformer_config = network_config['transformer']
    dim = transformer_config['dim']
    template_shape = transformer_config['backbone']['template']['shape']
    pseudo_data_generator = SiamFCPseudoDataGenerator(network_data_config['template_size'], network_data_config['search_size'], template_shape, dim)
    event_register.register_device_changed_hook(pseudo_data_generator)
    return pseudo_data_generator
