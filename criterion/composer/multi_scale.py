def _sum_compose(losses):
    loss = sum(v[0] for v in losses)
    loss_value = sum(v[1] for v in losses)
    loss_dict = {}
    [loss_dict.update(v[2]) for v in losses]
    return loss, loss_value, loss_dict


class MultiScaleLossComposer:
    def __init__(self, compose_fn, single_scale_loss_composers):
        self.single_scale_loss_composers = single_scale_loss_composers
        self.compose_fn = compose_fn

    def __call__(self, losses):
        return self.compose_fn(tuple(composer(loss) for loss, composer in zip(losses, self.single_scale_loss_composers)))

    def state_dict(self):
        state = []
        for single_scale_loss_composer in self.single_scale_loss_composers:
            state.append(single_scale_loss_composer.state_dict())
        return state

    def load_state_dict(self, states):
        for state, single_scale_loss_composer in zip(states, self.single_scale_loss_composers):
            single_scale_loss_composer.load_state_dict(state)

    def on_iteration_end(self, is_training):
        for single_scale_loss_composer in self.single_scale_loss_composers:
            single_scale_loss_composer.on_iteration_end(is_training)

    def on_epoch_begin(self, epoch):
        for single_scale_loss_composer in self.single_scale_loss_composers:
            single_scale_loss_composer.on_epoch_begin(epoch)
