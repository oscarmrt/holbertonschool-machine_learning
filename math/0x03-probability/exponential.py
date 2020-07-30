#!/usr/bin/env python3
"""Class Exponential that represents an exponential distribution"""


class Exponential():
    """Class Exponential"""
    e = 2.7182818285

    def __init__(self, data=None, lambtha=1.):
        """Class contructor"""
        if data is None:
            if lambtha <= 0:
                raise ValueError('lambtha must be a positive value')
            self.lambtha = float(lambtha)
        else:
            if type(data) is not list:
                raise TypeError('data must be a list')
            elif len(data) < 2:
                raise ValueError('data must contain multiple values')
            else:
                self.lambtha = (1 / (sum(data) / len(data)))

    def pdf(self, x):
        """Calculates the value of the PDF for a given time period"""
        if x < 0:
            return 0
        pdfP = self.lambtha * Exponential.e ** (-(x * self.lambtha))
        return pdfP

    def cdf(self, x):
        """Calculates the value of the CDF for a given time period"""
        if x < 0:
            return 0
        cdfP = 1 - Exponential.e ** (-(x * self.lambtha))
        return cdfP
