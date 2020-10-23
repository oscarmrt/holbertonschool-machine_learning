#!/usr/bin/env python3
"""Program that calculates the cofactor matrix of a matrix"""


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


def cofactor(matrix):
    """Function that calculates the cofactor matrix of a matrix"""
    if type(matrix) is not list or not matrix:
        raise TypeError("matrix must be a list of lists")
    for data in matrix:
        if type(data) is not list:
            raise TypeError("matrix must be a list of lists")
        if len(matrix) != len(data):
            raise ValueError("matrix must be a non-empty square matrix")
    if len(matrix) == 1:
        return [[1]]
    minor_matrix = [x[:] for x in matrix]
    for i in range(len(matrix)):
        sub = matrix[:i] + matrix[i + 1:]
        for j in range(len(matrix)):
            tmp = sub[:]
            for k in range(len(tmp)):
                tmp[k] = tmp[k][0:j] + tmp[k][j + 1:]
            if len(tmp) > 1:
                if len(tmp) > 2:
                    minor_matrix[i][j] = ((-1)**(i + j)) * determinant(tmp)
                else:
                    a = tmp[0][0]
                    b = tmp[0][1]
                    c = tmp[1][0]
                    d = tmp[1][1]
                    minor_matrix[i][j] = ((-1)**(i + j)) * (a * d - b * c)
            else:
                minor_matrix[i][j] = ((-1)**(i + j)) * tmp[0][0]
    return minor_matrix
