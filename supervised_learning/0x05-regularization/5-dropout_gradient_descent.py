#!/usr/bin/env python3
"""Program that updates the weights of a neural network
with Dropout regularization using gradient descent."""
import numpy as np


def dropout_gradient_descent(Y, weights, cache, alpha, keep_prob, L):
    """Function that updates the weights of a neural network
    with Dropout regularization using gradient descent."""
    m = Y.shape[1]
    dAl = cache['A' + str(L)] - Y
    for i in reversed(range(1, L + 1)):
        W_key = 'W' + str(i)
        b_key = 'b' + str(i)
        A_key = 'A' + str(i - 1)
        D_key = 'D' + str(i - 1)
        dw = (np.matmul(cache[A_key], dAl.T) / m)
        db = (np.sum(dAl, axis=1, keepdims=True) / m)
        if i - 1 > 0:
            dAl = np.matmul(weights['W' + str(
                i)].T, dAl) * (1 - (cache[A_key] ** 2)) * (cache[D_key]
                                                           / keep_prob)
        weights[W_key] = weights['W' + str(
            i)] - (alpha * dw).T
        weights[b_key] = weights['b' + str(
            i)] - (alpha * db)
