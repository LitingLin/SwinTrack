from enum import IntFlag, auto


class DataSplit(IntFlag):
    Training = auto()
    Validation = auto()
    Testing = auto()
    Challenge = auto()
    Full = auto()

    def __str__(self):
        if self.value == DataSplit.Full:
            return 'full'
        string = ''
        if self.value & DataSplit.Training:
            string += 'train'
        if self.value & DataSplit.Validation:
            string += 'val'
        if self.value & DataSplit.Testing:
            string += 'test'
        if self.value & DataSplit.Challenge:
            string += 'challenge'
        return string
