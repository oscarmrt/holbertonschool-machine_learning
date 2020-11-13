#!/usr/bin/env python3
"""Program that determines the probability of a markov chain being
in a particular state after a specified number of iterations"""
import numpy as np


def markov_chain(P, s, t=1):
    """Program that determines the probability of a markov chain being
    in a particular state after a specified number of iterations"""
    if type(P) is not np.ndarray or len(P.shape) != 2:
        return None
    if type(s) is not np.ndarray or len(s.shape) != 2:
        return None
    if P.shape[0] != P.shape[1]:
        return None
    if P.shape[1] != s.shape[1]:
        return None
    if t <= 0:
        return None
    pState = np.dot(s, np.linalg.matrix_power(P, t))
    return pState
