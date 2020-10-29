#!/usr/bin/env python3
"""Program that performs PCA on a dataset"""
import numpy as np


def pca(X, var=0.95):
    """Function that performs PCA on a dataset"""
    u, s, vh = np.linalg.svd(X)
    cumulative = np.cumsum(s) / sum(s)
    index = np.where(cumulative <= var, 1, 0)
    n = sum(index) + 1
    W = vh.T[:, :n]
    return W
