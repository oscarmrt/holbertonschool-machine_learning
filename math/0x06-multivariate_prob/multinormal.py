#!/usr/bin/env python3
"""Program that represents a Multivariate Normal distribution"""
import numpy as np


class MultiNormal:
    """Class that represents a Multivariate Normal distribution"""

    def __init__(self, data):
        """class constructor"""
        if not isinstance(data, np.ndarray) or len(data.shape) != 2:
            raise TypeError("data must be a 2D numpy.ndarray")
        if data.shape[0] < 2:
            raise ValueError("data must contain multiple data points")
        d = data.shape[0]
        n = data.shape[1]
        self.mean = np.mean(data, axis=1).reshape(d, 1)
        deviation = np.tile(self.mean.reshape(-1), n).reshape(n, d)
        cov = data.T - deviation
        self.cov = np.matmul(cov.T, cov)/(n - 1)
