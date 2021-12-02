import numpy as np


class DatasetsIndexingHelper:
    def __init__(self, dataset_length_list):
        self.dataset_length_list = dataset_length_list
        self.length = sum(dataset_length_list)

    def __getitem__(self, index: int):
        for index_of_datasets, length in enumerate(self.dataset_length_list):
            if index < length:
                return index_of_datasets, index
            index -= length
        raise IndexError

    def __len__(self):
        return self.length


class ApiGatewayRunThroughSamplerServerHandler:
    def __init__(self, datasets, seed):
        self.datasets_indexing_helper = DatasetsIndexingHelper(tuple(len(dataset) for dataset in datasets))

        self.seed = seed
        self.reset()

    def reset(self):
        rng_engine = np.random.Generator(np.random.PCG64(self.seed))
        indices = np.arange(len(self.datasets_indexing_helper))
        rng_engine.shuffle(indices)
        self.indices = indices
        self.position = 0
        self.done = 0
        self.stop_iteration = None
        self.client_worker_stop_flags = {}
        self.global_max_iteration = 0

    def __call__(self, command, response):
        if command[0] == 'get_next':
            worker_local_rank = command[1]
            stop_flag = False
            if worker_local_rank in self.client_worker_stop_flags:
                stop_flag = self.client_worker_stop_flags[worker_local_rank]

            if self.position < len(self.datasets_indexing_helper):
                index = self.indices[self.position]
                index_of_dataset, index_of_sequence = self.datasets_indexing_helper[index]
                response.set_body((index_of_dataset, index_of_sequence, stop_flag))
                self.position += 1
            else:
                response.set_body((None, None, stop_flag))
        elif command[0] == 'mark_done_and_get_status':
            client_local_rank = command[1]
            client_iteration_index = command[2]
            num_done = command[3]

            if self.global_max_iteration < client_iteration_index:
                self.global_max_iteration = client_iteration_index

            if client_local_rank not in self.client_worker_stop_flags:
                self.client_worker_stop_flags[client_local_rank] = False

            self.done += num_done
            assert self.done <= len(self.datasets_indexing_helper)
            is_done = self.done == len(self.datasets_indexing_helper)
            if is_done and self.stop_iteration is None:
                self.stop_iteration = self.global_max_iteration + 2

            self.client_worker_stop_flags[client_local_rank] = False
            if self.stop_iteration is not None:
                assert client_iteration_index < self.stop_iteration
                if client_iteration_index + 1 == self.stop_iteration:
                    self.client_worker_stop_flags[client_local_rank] = True

            response.set_body((self.stop_iteration, self.position, self.done, len(self.datasets_indexing_helper)))
        elif command[0] == 'reset':
            self.reset()
            response.set_body('ok')
        else:
            raise Exception(f'Unknown command received {command}')
