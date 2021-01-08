#!/usr/bin/env python3
"""Program that calculates the positional encoding for a transformer"""
import numpy as np


def get_angles(pos, i, d_model):
    """Function that get the angles"""
    angle_rates = 1 / np.power(10000, (2 * (i//2)) / np.float32(d_model))
    return pos * angle_rates

def positional_encoding(max_seq_len, dm):
    """Function that calculates the positional encoding for a transformer"""
    angle_rads = get_angles(np.arange(max_seq_len)[:, np.newaxis],
                            np.arange(dm)[np.newaxis, :],
                            dm)
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
    return angle_rads
