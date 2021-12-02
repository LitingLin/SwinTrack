def seed_all_rng(seed=0):
    import random
    random.seed(seed)
    import numpy as np
    np.random.seed(random.randint(0, 255))
    import torch
    torch.manual_seed(random.randint(0, 255))


def enable_deterministic_computation():
    import torch
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
