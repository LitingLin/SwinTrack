from miscellanies.simple_api_gateway import ServerLauncher, Client, CallbackFactory
from data.tracking.sampler._sampling_algos.sequence_picking.run_through._server import ApiGatewayRunThroughSamplerServerHandler


class RunThroughSequencePickingOrchestrationServer:
    def __init__(self, datasets, socket_address, seed: int):
        self.server_callback = CallbackFactory(ApiGatewayRunThroughSamplerServerHandler, (datasets, seed))
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
            self.client.stop()
            self.server.stop()

    def reset(self):
        if self.server.is_launched():
            assert self.client('reset', ) == 'ok'


class RunThroughSequencePickingClient:
    def __init__(self, socket_address, rank):
        self.client = Client(socket_address)
        if rank is None:
            rank = 0
        self.rank = rank

    def get_next(self):
        '''
        normally
            return index_of_dataset, index_of_sequence
        or sequences is exhausted
            return None
        '''
        index_of_dataset, index_of_sequence, is_done = self.client('get_next', self.rank)

        if is_done:
            raise StopIteration
        if index_of_dataset is None:
            return None
        return index_of_dataset, index_of_sequence

    def mark_done_and_get_status(self, index_iteration, num):
        return self.client('mark_done_and_get_status', self.rank, index_iteration, num)

    def start(self):
        self.client.start()

    def stop(self):
        self.client.stop()
