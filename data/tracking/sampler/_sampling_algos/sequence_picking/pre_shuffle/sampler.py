import numpy as np
import copy


class PreShuffledSequencePicker:
    def __init__(self, datasets, datasets_sampling_probability, total_size, seed):
        sizes = datasets_sampling_probability * total_size
        sizes = sizes.astype(np.int)
        sizes[-1] = total_size - sum(sizes[:len(sizes) - 1])

        rng_engine = np.random.Generator(np.random.PCG64(seed))

        dataset_indices = []
        sequence_indices = []
        for index, (dataset, size) in enumerate(zip(datasets, sizes)):
            current_dataset_indices = []
            indices = np.arange(len(dataset))
            current_size = 0
            while current_size < size:
                current_indices = copy.copy(indices)
                rng_engine.shuffle(current_indices)
                current_dataset_indices.append(current_indices)
                current_size += len(indices)
            current_dataset_indices = np.concatenate(current_dataset_indices)
            current_dataset_indices = current_dataset_indices[: size]
            dataset_indices.append(np.full(size, index))
            sequence_indices.append(current_dataset_indices)
        dataset_indices = np.concatenate(dataset_indices)
        sequence_indices = np.concatenate(sequence_indices)
        shuffle_indices = rng_engine.permutation(len(dataset_indices))
        dataset_indices = dataset_indices[shuffle_indices]
        sequence_indices = sequence_indices[shuffle_indices]
        self.dataset_indices = dataset_indices
        self.sequence_indices = sequence_indices

    def __getitem__(self, index):
        return self.dataset_indices[index], self.sequence_indices[index]

    def __len__(self):
        return len(self.dataset_indices)
