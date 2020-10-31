#!/usr/bin/env python3
"""Program that calculates the posterior probability that the
probability of developing severe side effects falls within a
specific range given the data:"""
from scipy import special


def posterior(x, n, p1, p2):
    """Function that calculates the posterior probability that the
    probability of developing severe side effects falls within a
    specific range given the data:"""
    if not isinstance(n, int) or n < 1:
        raise ValueError("n must be a positive integer")
    if type(x) != int or x < 0:
        err = "x must be an integer that is greater than or equal to 0"
        raise ValueError(err)
    if x > n:
        raise ValueError("x cannot be greater than n")
    if not isinstance(P, np.ndarray) or len(P.shape) != 1:
        raise TypeError("P must be a 1D numpy.ndarray")
    if not isinstance(p1, float) or p1 < 0 or p1 > 1:
        raise TypeError("p1 must be a float in the range [0, 1]")
    if not isinstance(p2, float) or p2 < 0 or p2 > 1:
        raise TypeError("p2 must be a float in the range [0, 1]")
    if p2 <= p1:
        raise ValueError("p2 must be greater than p1")
    return p1
