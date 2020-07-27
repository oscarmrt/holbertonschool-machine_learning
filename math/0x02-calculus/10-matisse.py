#!/usr/bin/env python3
"""Program that calculates the derivative of a polynomial"""


def poly_derivative(poly):
    """Function that calculates the derivative of a polynomial"""
    if type(poly) is not list or len(poly) == 0:
        return None
    elif len(poly) == 1:
        return [0]
    else:
        dvPoly = [poly[x] * x for x in range(1, len(poly))]
        if dvPoly == 0:
            return [0]
        return dvPoly
