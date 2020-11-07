#!/usr/bin/env python3
"""Program that calculates the probability density
function of a Gaussian distribution"""
import numpy as np


def pdf(X, m, S):
    """Function that calculates the probability density
    function of a Gaussian distribution"""
    if type(X) is not np.ndarray or len(X.shape) != 2:
        return None
    if type(m) is not np.ndarray or len(m.shape) != 1:
        return None
    if type(S) is not np.ndarray or len(S.shape) != 2:
        return None
    n, d = X.shape
    if d != m.shape[0] or (d, d) != S.shape:
        return None
    inv = np.linalg.inv(S)
    det = np.linalg.det(S)
    a = 1 / np.sqrt((((2 * np.pi) ** (d) * det)))
    inv = np.einsum('...k,kl,...l->...', (X - m), inv, (X - m))
    b = np.exp(-(1 / 2) * inv)
    P = a * b
    P = np.where(P >= 1e-300, P, 1e-300)
    return P
