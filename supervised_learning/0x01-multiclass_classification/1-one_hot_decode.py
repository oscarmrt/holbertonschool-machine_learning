#!/usr/bin/env python3
"""Program that converts a one-hot matrix into a vector of labels"""
import numpy as np


def one_hot_decode(one_hot):
    """Function that converts a one-hot matrix into a vector of labels"""
    if len(one_hot) == 0:
        return None
    elif len(one_hot.shape) != 2:
        return None
    elif not isinstance(one_hot, np.ndarray):
        return None
    else:
        return np.argmax(one_hot, axis=0)
