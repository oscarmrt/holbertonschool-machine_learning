#!/usr/bin/env python3
"""Program that updates a variable using the gradient
descent with momentum optimization algorithm"""
import numpy as np


def update_variables_momentum(alpha, beta1, var, grad, v):
    """Function that updates a variable using the gradient
    descent with momentum optimization algorithm"""
    V = (beta1 * v) + ((1 - beta1) * grad)
    W = var - (alpha * V)
    return W, V
