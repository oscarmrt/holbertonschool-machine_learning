#!/usr/bin/env python3
"""Program that updates the weights and biases of a neural network
using gradient descent with L2 regularization."""
import numpy as np


def l2_reg_gradient_descent(Y, weights, cache, alpha, lambtha, L):
    """Function that updates the weights and biases of a neural network
    using gradient descent with L2 regularization."""
    m = len(Y[1])
    dz = cache['A'+str(L)] - Y
    for i in range(L, 0, -1):
        L2Cost = (lambtha / m) * weights['W'+str(i)]
        db = (1 / m) * np.sum(dz, axis=1, keepdims=True)
        dW = ((1 / m) * np.matmul(dz, cache['A'+str(i-1)].T)) + L2Cost
        dz = np.matmul(weights['W'+str(i)].T, dz) *\
            ((1 - cache['A'+str(i-1)] ** 2))
        weights['W'+str(i)] = weights['W'+str(i)] -\
            (alpha * dW)
        weights['b'+str(i)] = weights['b'+str(i)] -\
            (alpha * db)
