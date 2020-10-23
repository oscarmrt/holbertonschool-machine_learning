#!/usr/bin/env python3
"""Program that calculates the definiteness of a matrix"""
import numpy as np


def definiteness(matrix):
    """Function that calculates the definiteness of a matrix"""
    if type(matrix) != np.ndarray:
        raise TypeError("matrix must be a numpy.ndarray")
    if len(matrix) == 0:
        return None
    for data in matrix:
        if len(data) != len(matrix):
            return None
    if not np.allclose(matrix, matrix.T):
        return None
    w, v = np.linalg.eig(matrix)
    positiveD = 0
    positiveSD = 0
    negativeD = 0
    negativeSD = 0
    for x in w:
        if x > 0:
            positiveD += 1
        if x >= 0:
            positiveSD += 1
        if x < 0:
            negativeD += 1
        if x <= 0:
            negativeSD += 1
    if positiveD == len(w):
        return "Positive definite"
    if positiveSD == len(w):
        return "Positive semi-definite"
    if negativeD == len(w):
        return "Negative definite"
    if negativeSD == len(w):
        return "Negative semi-definite"
    if positiveD and negativeD:
        return "Indefinite"
    else:
        return None
