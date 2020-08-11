#!/usr/bin/env python3
"""Program that converts a numeric label vector into a one-hot matrix"""
import numpy as np


def one_hot_encode(Y, classes):
    """Function that converts a numeric label vector into a one-hot matrix"""
    if len(Y) == 0 or type(classes) is not int:
        return None
    elif classes <= np.argmax(Y):
        return None
    else:
        onehot_encoder = np.zeros((classes, len(Y)))
        onehot_encoder[Y, np.arange(len(Y))] = 1
        return onehot_encoder
