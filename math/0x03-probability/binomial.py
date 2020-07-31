#!/usr/bin/env python3
"""Class Binomial that represents a binomial distribution"""


class Binomial():
    """Class Binomial"""
    e = 2.7182818285
    pi = 3.1415926536

    def __init__(self, data=None, n=1, p=0.5):
        """Class constructor"""
        if data is None:
            if n <= 0:
                raise ValueError('n must be a positive value')
            if p < 0 or p > 1:
                raise ValueError('p must be greater than 0 and less than 1')
            self.n = int(n)
            self.p = float(p)
        else:
            if type(data) is not list:
                raise TypeError('data must be a list')
            if len(data) < 2:
                raise ValueError('data must contain multiple values')
            else:
                mean = sum(data) / len(data)
                variance = sum([((x - mean) ** 2) for x in data]) / len(data)
                self.p = 1 - variance / mean
                self.n = int(round(mean / self.p))
                self.p = float(mean / self.n)

    def pmf(self, k):
        """Calculates the value of the PMF for a given number of “successes”"""
        if k < 0:
            return 0
        k = int(k)
        n_f = self.factorial(self.n)
        k_f = self.factorial(k)
        nk_f = self.factorial(self.n - k)
        pk = self.p ** k
        pmf = (n_f / (k_f * (nk_f))) * pk * ((1 - self.p) ** (self.n - k))
        return pmf

    def cdf(self, k):
        """Calculates the value of the CDF for a given number of “successes”"""
        if k < 0:
            return 0
        k = int(k)
        cdf = sum(self.pmf(i) for i in range(k + 1))
        return cdf

    def factorial(self, x):
        """calculates factorial"""
        if x == 0:
            return 1
        fact = 1
        for i in range(1, int(x) + 1):
            fact = fact * i
        return fact
