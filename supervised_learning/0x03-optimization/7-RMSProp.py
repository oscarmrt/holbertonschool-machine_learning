#!/usr/bin/env python3
"""Program that updates a variable
using the RMSProp optimization algorithm"""
import numpy as np


def update_variables_RMSProp(alpha, beta2, epsilon, var, grad, s):
    """Program that updates a variable using the
    RMSProp optimization algorithm"""
    sdv = beta2 * s + (1 - beta2) * (grad ** 2)
    var = var - alpha * (grad / (np.sqrt(sdv) + epsilon))
    return var, sdv
