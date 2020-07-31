#!/usr/bin/env python3
"""Class Normal that represents an normal distribution"""


class Normal():
    """Class Normal"""
    e = 2.7182818285
    pi = 3.1415926536

    def __init__(self, data=None, mean=0., stddev=1.):
        """Class constructor"""
        if data is None:
            if stddev <= 0:
                raise ValueError('stddev must be a positive value')
            self.stddev = float(stddev)
            self.mean = float(mean)
        else:
            if type(data) is not list:
                raise TypeError('data must be a list')
            elif len(data) < 2:
                raise ValueError('data must contain multiple values')
            else:
                self.mean = sum(data) / len(data)
                self.stddev = (sum([(x - self.mean) **
                               2 for x in data]) / (len(data))) ** .5

    def z_score(self, x):
        """Calculates the z-score of a given x-value"""
        return (x - self.mean) / self.stddev

    def x_value(self, z):
        """Calculates the z-score of a given z-value"""
        return (z * self.stddev) + self.mean

    def pdf(self, x):
        """Calculates the value of the PDF for a given x-value"""
        fctr = float(1 / ((self.stddev) * (2 * Normal.pi) ** .5))
        exp = -.5 * (((x - self.mean) / self.stddev) ** 2)
        return float(fctr * (Normal.e ** exp))
