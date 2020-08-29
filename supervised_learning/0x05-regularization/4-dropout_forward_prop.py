#!/usr/bin/env python3
"""Program  that conducts forward propagation using Dropout"""
import numpy as np


def dropout_forward_prop(X, weights, L, keep_prob):
    """Function  that conducts forward propagation using Dropout"""
    cache = {}
    for i in range(L):
        m = 'D' + str(i)
        if i > 0:
            cache[m] = (np.random.rand(X.shape[0],
                                       X.shape[1])
                        < keep_prob).astype(int)
            X = np.multiply(X, cache[m])
            X = X / keep_prob
        cache["A" + str(i)] = X
        Z = np.dot(weights["W" + str(i + 1)], X) + weights["b" + str(i + 1)]
        if i == L - 1:
            t = np.exp(Z)
            X = t / np.sum(t, axis=0, keepdims=True)
            cache["A" + str(i + 1)] = X
        else:
            X = np.sinh(Z) / np.cosh(Z)
    return cache
