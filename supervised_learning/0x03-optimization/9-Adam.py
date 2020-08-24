#!/usr/bin/env python3
"""Program that updates a variable in place using
the Adam optimization algorithm"""
import numpy as np


def update_variables_Adam(alpha, beta1, beta2, epsilon, var, grad, v, s, t):
    """Function that updates a variable in place using
    the Adam optimization algorithm"""
    V = (beta1 * v) + ((1 - beta1) * grad)
    sdv = beta2 * s + (1 - beta2) * (grad ** 2)
    vdvA = V / (1 - (beta1 ** t))
    sdvA = sdv / (1 - (beta2 ** t))
    var = var - alpha * vdvA / (np.sqrt(sdvA) + epsilon)
    return var, V, sdv
