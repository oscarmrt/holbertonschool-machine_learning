#!/usr/bin/env python3
"""Program that calculates the mean
and covariance of a data set"""
import numpy as np


def mean_cov(X):
    """Function that calculates the
    mean and covariance of a data set"""
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        raise TypeError("X must be a 2D numpy.ndarray")
    if X.shape[0] < 2:
        raise ValueError("X must contain multiple data points")
    n = X.shape[0]
    d = X.shape[1]
    mean = np.mean(X, axis=0).reshape(1, d)
    deviation = np.tile(mean, n).reshape(n, d)
    cov = X - deviation
    cov = np.matmul(cov.T, cov)/(n - 1)
    return mean, cov
