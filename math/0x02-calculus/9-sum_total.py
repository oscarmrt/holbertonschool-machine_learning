#!/usr/bin/env python3
"""Program that calculates \\sum_{i=1}^{n} i^2"""


def summation_i_squared(n):
    """Function that calculates \\sum_{i=1}^{n} i^2"""
    if type(n) is not int or n < 1:
        return None
    return int((n * (1 + n)) * ((2 * n) + 1) / 6)
