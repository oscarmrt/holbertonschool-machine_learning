#!/usr/bin/env python3
"""Class Poisson that represents a poisson distribution"""


class Poisson():
    """Class Poison"""
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
                self.lambtha = (sum(data) / len(data))

    def pmf(self, k):
        """Calculates the value of the PMF for a given number of “successes”"""
        factK = 1
        k = int(k)
        if k < 0:
            return 0
        for x in range(1, k + 1):
            factK *= x
        pmfP = (Poisson.e ** (-self.lambtha)) * (self.lambtha ** k) / factK
        return pmfP
