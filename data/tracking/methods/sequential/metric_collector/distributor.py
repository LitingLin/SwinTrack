import re


class SequenceDistributor:
    def __init__(self, rules):
        self.rules = tuple(re.compile(rule) for rule in rules)

    def __call__(self, collected_sequences, sub_collectors, io_thread):
        collated_sequences = [None] * len(sub_collectors)
        for collected_sequence in collected_sequences:
            dataset_unique_id = collected_sequence[0]
            dispatched = False
            for index, rule in enumerate(self.rules):
                if rule.search(dataset_unique_id) is not None:
                    if collated_sequences[index] is None:
                        collated_sequences[index] = []
                    collated_sequences[index].append(collected_sequence)
                    dispatched = True
                    break
            assert dispatched, "No rule matched the dataset unique id: {}".format(dataset_unique_id)

        for collated_sequence, sub_collector in zip(collated_sequences, sub_collectors):
            if collated_sequence is not None:
                sub_collector.accept_evaluated_sequence(collated_sequence, io_thread)
