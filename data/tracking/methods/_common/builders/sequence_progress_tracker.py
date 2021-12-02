from core.run.metric_logger.context import get_logger


class SequenceProcessTracking:
    def __init__(self, dataloader, run_through_sequence_picker):
        self.dataloader = dataloader
        self.run_through_sequence_picker = run_through_sequence_picker

    def __iter__(self):
        self.dataloader_iter = iter(self.dataloader)
        self.index = 0
        self.stop_iteration = None
        return self

    def __len__(self):
        return len(self.dataloader)

    def __next__(self):
        if self.stop_iteration is None or self.index < self.stop_iteration:
            return next(self.dataloader_iter)
        else:
            while True:
                next(self.dataloader_iter)

    def get_metric_definitions(self):
        return {'local': [
            {'name': 'scheduled', 'window_size': 1, 'fmt': '{value}'},
            {'name': 'done', 'window_size': 1, 'fmt': '{value}'},
            {'name': 'total', 'window_size': 1, 'fmt': '{value}'},
        ]}

    def post_processing(self, outputs):
        num_evaluated_sequences = 0
        if outputs is not None:
            num_evaluated_sequences = len(outputs)
        self.stop_iteration, sequence_position, sequence_done, total_sequences = self.run_through_sequence_picker.mark_done_and_get_status(self.index, num_evaluated_sequences)
        self.index += 1
        get_logger().log(local={'scheduled': sequence_position, 'done': sequence_done, 'total': total_sequences})
        return outputs
