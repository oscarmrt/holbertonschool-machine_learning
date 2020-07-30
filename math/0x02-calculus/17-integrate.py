#!/usr/bin/env python3
"""Program that calculates the integral of a polynomial"""


def poly_integral(poly, C=0):
    """Function that calculates the integral of a polynomial"""
    if type(poly) is not list or len(poly) == 0 or type(C) is not int:
        return None
    elif poly == [0]:
        return [C]
    else:
        integralPoly = [C]
        for x in range(len(poly)):
            y = poly[x] / (x + 1)
            if y.is_integer():
                integralPoly.append(int(y))
            else:
                integralPoly.append(y)
        return integralPoly
