#!/usr/bin/env python3
"""Program that calculates the determinant of a matrix"""


def determinant(matrix):
    """Function that calculates the determinant of a matrix"""
    if type(matrix) is not list or not matrix:
        raise TypeError('matrix must be a list of lists')
    for data in matrix:
        if type(data) is not list:
            raise TypeError('matrix must be a list of lists')
    if len(matrix) == 1 and len(matrix[0]) == 0:
        return 1
    for i in range(len(matrix)):
        if len(matrix) != len(matrix[i]):
            raise ValueError('matrix must be a square matrix')
    n = len(matrix)
    AM = matrix[:]
    for fd in range(n):
        for i in range(fd+1, n):
            if AM[fd][fd] == 0:
                AM[fd][fd] == 1.0e-18
            crScaler = AM[i][fd] / AM[fd][fd]
            for j in range(n):
                AM[i][j] = AM[i][j] - crScaler * AM[fd][j]
    product = 1
    for i in range(n):
        product *= AM[i][i]
    return round(product)
