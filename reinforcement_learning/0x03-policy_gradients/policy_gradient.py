#!/usr/bin/env python3
"""Program that computes to policy with a weight of a matrix"""
import numpy as np


def policy(matrix, weight):
    """Function that computes to policy with a weight of a matrix"""
    z = matrix.dot(weight)
    exp = np.exp(z)
    return exp / exp.sum()
