#!/usr/bin/env python3
"""Program that determines the steady state probabilities
of a regular markov chain"""
import numpy as np


def regular(P):
    """Function that determines the steady state probabilities
    of a regular markov chain"""
    if type(P) is not np.ndarray or len(P.shape) != 2:
        return None
    if P.shape[0] != P.shape[1]:
        return None
    m = ((len(P.shape) - 1) ** 2) + 1
    T = P.copy()
    for i in range(m):
        T = np.linalg.matrix_power(T, 2)
        if np.any(T <= 0):
            return None
    for i in range(10):
        P = np.linalg.matrix_power(P, 2)
    reg = np.array([P[0]])
    return reg
