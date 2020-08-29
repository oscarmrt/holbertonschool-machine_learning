#!/usr/bin/env python3
"""Program that updates the weights of a neural network
with Dropout regularization using gradient descent."""
import numpy as np


def dropout_gradient_descent(Y, weights, cache, alpha, keep_prob, L):
    """Function that updates the weights of a neural network
    with Dropout regularization using gradient descent."""
    m = len(Y[1])
    dz = cache['A'+str(L)] - Y
    for i in range(L, 0, -1):
        db = (1 / m) * np.sum(dz, axis=1, keepdims=True)
        dW = ((1 / m) * np.matmul(dz, cache['A'+str(i-1)].T))
        if i - 1 > 0:
            dz = np.matmul(weights['W'+str(i)].T, dz) *\
                ((1 - cache['A'+str(i-1)] ** 2)) *\
                (cache['D'+str(i-1)] / keep_prob)
            weights['W'+str(i)] = weights['W'+str(i)] -\
                (alpha * dW)
            weights['b'+str(i)] = weights['b'+str(i)] -\
                (alpha * db)