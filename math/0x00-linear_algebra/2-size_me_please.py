#!/usr/bin/env python3
"""Program that calculates the shape
of a matrix"""


def matrix_shape(matrix):
    """Function that calculates the shape
    of a matrix"""
    mShape = []
    while type(matrix) == list:
        mShape.append(len(matrix))
        if type(matrix[0]) == list:
            matrix = matrix[0]
        else:
            break
    return mShape
