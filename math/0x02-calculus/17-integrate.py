#!/usr/bin/env python3
"""Program that calculates the integral of a polynomial"""


def poly_integral(poly, C=0):
    """Function that calculates the integral of a polynomial"""
    if type(poly) is not list or len(poly) == 0 or type(C) is None:
        return None
