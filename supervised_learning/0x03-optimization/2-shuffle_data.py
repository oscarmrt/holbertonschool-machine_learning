#!/usr/bin/env python3
"""Program that shuffles the data points in two matrices the same way"""
import numpy as np


def shuffle_data(X, Y):
    """Program that shuffles the data points in
    two matrices the same way"""
    rdm = np.random.permutation(X.shape[0])
    mtx_x = X[rdm]
    mtx_y = Y[rdm]
    return mtx_x, mtx_y
