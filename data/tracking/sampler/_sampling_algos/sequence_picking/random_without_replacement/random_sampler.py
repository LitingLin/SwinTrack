from miscellanies.simple_api_gateway import ServerLauncher, Client
from data.tracking.sampler._sampling_algos.sequence_picking.random_without_replacement._server import ApiGatewayRandomSamplerServerHandler


class RandomSequencePickingOrchestrationServer:
    def __init__(self, datasets, datasets_sampling_probability, socket_address, seed: int):
        self.server_callback = ApiGatewayRandomSamplerServerHandler(datasets, datasets_sampling_probability, seed)
        self.server = ServerLauncher(socket_address, self.server_callback)
        self.client = Client(socket_address)

    def __del__(self):
        self.client.stop()
        self.server.stop()

    def start(self):
        if not self.server.is_launched():
            self.server.launch()
            self.client.start()

    def stop(self):
        if self.server.is_launched():
            self.server_callback.set_state(self.client('get_state', ))
            self.client.stop()
            self.server.stop()

    def state_dict(self):
        if self.server.is_launched():
            state = self.client('get_state', )
            self.server_callback.set_state(state)
            return state
        else:
            return self.server_callback.get_state()

    def load_state_dict(self, state):
        if self.server.is_launched():
            self.client('set_state', state)
            self.server_callback.set_state(state)
        else:
            self.server_callback.set_state(state)

    def get_status(self):
        return self.client('get_status', )


class RandomSequencePickingClient:
    def __init__(self, socket_address):
        self.client = Client(socket_address)

    def get_next(self):
        index_of_dataset, index_of_sequence = self.client('get_next', )
        return index_of_dataset, index_of_sequence

    def start(self):
        self.client.start()

    def stop(self):
        self.client.stop()
