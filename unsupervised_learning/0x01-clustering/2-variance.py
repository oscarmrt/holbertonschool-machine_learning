#!/usr/bin/env python3
"""Program that calculates the total intra-cluster
variance for a data set"""
import numpy as np


def variance(X, C):
    """Function that calculates the total intra-cluster
    variance for a data set"""
    if type(X) is not np.ndarray or len(X.shape) != 2:
        return None
    if type(C) is not np.ndarray or len(C.shape) != 2:
        return None
    if X.shape[0] < C.shape[0] or X.shape[1] != C.shape[1]:
        return None
    distances = np.linalg.norm(X - C[:, np.newaxis], axis=2)
    clss = np.min(distances, axis=0)
    var = (clss**2).sum()
    return var.sum()
