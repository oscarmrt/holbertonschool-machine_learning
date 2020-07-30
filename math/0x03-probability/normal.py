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
