import numpy as np


def gaussian(x: np.ndarray, mu, sig):
    return np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))


def get_sig(value, x, mu):  # positive only
    return np.sqrt(-(np.power((x - mu), 2.) / (2. * np.log(value))))
