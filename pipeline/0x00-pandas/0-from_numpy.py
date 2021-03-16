#!/usr/bin/env python3
"""Program that creates a pd.DataFrame from a np.ndarray"""
import numpy as np
import pandas as pd


def from_numpy(array):
    """Function that creates a pd.DataFrame from a np.ndarray"""
    idx = [chr(i+65) for i in range(array.shape[1])]
    return pd.DataFrame(array, columns=idx)
