#!/usr/bin/env python3
"""Program that calculates the likelihood of obtaining
this data given various hypothetical probabilities of
developing severe side effects"""
import numpy as np


def likelihood(x, n, P):
    """Function that calculates the likelihood of obtaining
    this data given various hypothetical probabilities of
    developing severe side effects"""
    if not isinstance(n, int) or n < 1:
        raise ValueError("n must be a positive integer")
    if type(x) != int or x < 0:
        err = "x must be an integer that is greater than or equal to 0"
        raise ValueError(err)
    if x > n:
        raise ValueError("x cannot be greater than n")
    if not isinstance(P, np.ndarray) or len(P.shape) != 1:
        raise TypeError("P must be a 1D numpy.ndarray")
    if np.amin(P) < 0 or np.amax(P) > 1:
        raise ValueError("All values in P must be in the range [0, 1]")
    fact = np.math.factorial
    tmp = fact(n) / (fact(x) * fact(n - x))
    lh = tmp * (P**x) * ((1 - P)**(n - x))
    return lh
