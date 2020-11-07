#!/usr/bin/env python3
"""Program that calculates the maximization step
in the EM algorithm for a GMM"""
import numpy as np


def maximization(X, g):
    """Program that calculates the maximization step
    in the EM algorithm for a GMM"""
    if type(X) is not np.ndarray or len(X.shape) != 2:
        return None, None, None
    if type(g) is not np.ndarray or len(g.shape) != 2:
        return None, None, None
    n, d = X.shape
    k, _ = g.shape
    if n != g.shape[1]:
        return None, None, None
    if not np.isclose(g.sum(), n):
        return None, None, None
    m_c = g.sum(axis=1)
    pi = 1 / n * m_c
    m = np.matmul(g, X)
    S = np.zeros((k, d, d))
    for i in range(k):
        m[i] /= m_c[i]
        S[i] = np.matmul(g[i].reshape(1, n) * (X - m[i]).T, (X - m[i]))
        S[i] /= m_c[i]
    return pi, m, S
