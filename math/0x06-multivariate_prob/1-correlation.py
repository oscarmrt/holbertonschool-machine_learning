#!/usr/bin/env python3
"""Program that calculates a correlation matrix"""
import numpy as np


def correlation(C):
    """Function that calculates a correlation matrix"""
    if not isinstance(C, np.ndarray):
        raise TypeError("C must be a numpy.ndarray")
    if C.shape[0] != C.shape[1]:
        raise ValueError("C must be a 2D square matrix")
    diagonal = np.diag(C)
    standard = np.sqrt(diagonal)
    outer = np.outer(standard, standard)
    correlation = C / outer
    return correlation
