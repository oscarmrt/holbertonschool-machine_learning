#!/usr/bin/env python3
"""Program that represents a Multivariate Normal distribution"""
import numpy as np


class MultiNormal:
    """Class that represents a Multivariate Normal distribution"""

    def __init__(self, data):
        """class constructor"""
        if not isinstance(data, np.ndarray) or len(data.shape) != 2:
            raise TypeError("data must be a 2D numpy.ndarray")
        if data.shape[1] < 2:
            raise ValueError("data must contain multiple data points")
        d = data.shape[0]
        n = data.shape[1]
        self.mean = np.mean(data, axis=1).reshape(d, 1)
        deviation = np.tile(self.mean.reshape(-1), n).reshape(n, d)
        cov = data.T - deviation
        self.cov = np.matmul(cov.T, cov)/(n - 1)

    def pdf(self, x):
        """public instance method that calculates the PDF at a data point"""
        if not isinstance(x, np.ndarray):
            raise TypeError("x must be a numpy.ndarray")
        d = self.mean.shape[0]
        if len(x.shape) != 2 or x.shape[1] != 1 or x.shape[0] != d:
            raise ValueError("x must have the shape ({}, 1)".format(d))
        inv = np.linalg.inv(self.cov)
        det = np.linalg.det(self.cov)
        a = 1 / np.sqrt((((2 * np.pi)**(x.shape[0]) * det)))
        inv = np.matmul((x - self.mean).T, inv)
        b = np.exp(-(1/2) * ((np.matmul(inv, (x - self.mean)))))
        pdf = a * b
        return pdf[0][0]
