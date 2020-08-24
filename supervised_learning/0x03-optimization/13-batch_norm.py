#!/usr/bin/env python3
"""Program that normalizes an unactivated output of
a neural network using batch normalization"""
import numpy as np


def batch_norm(Z, gamma, beta, epsilon):
    """Function that normalizes an unactivated output of
    a neural network using batch normalization"""
    mean = np.sum(Z, axis=0) / Z.shape[0]
    var = np.sum(np.power(Z - mean, 2), axis=0) / Z.shape[0]
    ZNormal = (Z - mean) / np.sqrt(var + epsilon)
    ZN = gamma * ZNormal + beta
    return ZN
