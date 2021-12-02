class SamplingAlgo_SequentialSampling:
    def __init__(self, length):
        assert length > 0
        self.position = -1
        self.length_ = length

    def __getstate__(self):
        return self.position, self.length_

    def __setstate__(self, state):
        position, length = state
        self.position = position
        self.length_ = length

    def move_next(self):
        self.position += 1
        return self.position < self.length_

    def current(self):
        if not 0 <= self.position < self.length_:
            raise IndexError
        return self.position

    def reset(self):
        self.position = -1

    def length(self):
        return self.length_
