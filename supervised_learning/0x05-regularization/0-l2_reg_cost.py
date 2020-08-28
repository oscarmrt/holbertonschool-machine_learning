#!/usr/bin/env python3
"""Program that calculates the cost of
a neural network with L2 regularization"""
import numpy as np


def l2_reg_cost(cost, lambtha, weights, L, m):
    """Program that calculates the cost of
    a neural network with L2 regularization"""
    nml = 0
    for x in range(1, L + 1):
        weight = weights['W' + str(x)]
        nml = nml + np.linalg.norm(weight)
    l2Cost = cost + (lambtha / (2 * m) * nml)
    return l2Cost
