from datasets.MOT.constructor.base_interface import MultipleObjectTrackingDatasetConstructor


def get_mot_class_definition():
    return {
        1: 'Pedestrian',
        2: 'Person on vehicle',
        3: 'Car',
        4: 'Bicycle',
        5: 'Motorbike',
        6: 'Non motorized vehicle',
        7: 'Static person',
        8: 'Distractor',
        9: 'Occluder',
        10: 'Occluder on the ground',
        11: 'Occluder full',
        12: 'Reflection',
        13: '(Unknown)'
    }


def get_mot20_sequences_from_path(sequences):
    valid_sequences = {}
    for sequence in sequences:
        words = sequence.split('-')
        assert len(words) == 2
        assert words[0] == 'MOT20'
        if words[1] not in valid_sequences:
            valid_sequences[words[1]] = sequence
    return valid_sequences.values()


def construct_MOT20(constructor: MultipleObjectTrackingDatasetConstructor, seed):
    from .MOT17 import construct_MOT
    construct_MOT(constructor, seed, get_mot20_sequences_from_path, get_mot_class_definition())
