#!/usr/bin/env python3
"""Program that tests for the optimum number of
clusters by variance"""
import numpy as np
kmeans = __import__('1-kmeans').kmeans
variance = __import__('2-variance').variance


def optimum_k(X, kmin=1, kmax=None, iterations=1000):
    """Program that tests for the optimum number of
    clusters by variance"""
    if type(X) is not np.ndarray or len(X.shape) != 2:
        return None, None
    if kmax is None:
        kmax = X.shape[0]
    if type(kmin) is not int or type(kmax) is not int:
        return None, None
    if kmin < 1 or kmax < 1:
        return None, None
    if kmin >= kmax:
        return None, None
    if type(iterations) is not int or iterations < 1:
        return None, None
    results = []
    d_vars = []
    for k in range(kmin, kmax + 1):
        C, clss = kmeans(X, k, iterations)
        results.append((C, clss))
        var = variance(X, results[0][0]) - variance(X, C)
        d_vars.append(var)
    return results, d_vars
