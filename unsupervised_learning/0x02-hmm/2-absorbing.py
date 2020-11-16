#!/usr/bin/env python3
"""Program that determines if a markov chain is absorbing"""
import numpy as np


def absorbing(P):
    """Function that determines if a markov chain is absorbing"""
    if type(P) is not np.ndarray or len(P.shape) != 2:
        return None
    if P.shape[0] != P.shape[1]:
        return None
    if np.any(P < 0):
        return None
    if np.min(P ** 2) < 0 or np.min(P ** 3) < 0:
        return None
    P = P.copy()
    absorb = np.ndarray(P.shape[0])
    while True:
        prev = absorb.copy()
        absorb = np.any(P == 1, axis=0)
        if absorb.all():
            return True
        if np.all(absorb == prev):
            return False
        absorbed = np.any(P[:, absorb], axis=1)
        P[absorbed, absorbed] = 1
