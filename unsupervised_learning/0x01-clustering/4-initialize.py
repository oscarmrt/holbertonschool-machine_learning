#!/usr/bin/env python3
"""Program that initializes variables for a
Gaussian Mixture Model"""
import numpy as np
kmeans = __import__('1-kmeans').kmeans


def initialize(X, k):
    """Function that initializes variables for a
    Gaussian Mixture Model"""
    if type(X) is not np.ndarray or len(X.shape) != 2:
        return None, None, None
    if type(k) is not int or k < 1:
        return None, None, None
    _, d = X.shape
    pi = np.ones((k)) / k
    m, clss = kmeans(X, k)
    S = np.zeros((k, d, d))
    S[:] = np.eye(d, d)
    return pi, m, S
