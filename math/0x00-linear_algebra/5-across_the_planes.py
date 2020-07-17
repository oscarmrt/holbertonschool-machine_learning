#!/usr/bin/env python3
"""Program that adds two matrices element-wise"""


def add_matrices2D(mat1, mat2):
    """Function that adds two matrices element-wise"""
    addMatrix = []
    if (len(mat1) == len(mat2)) and (len(mat1[0]) == len(mat2[0])):
        for x in range(len(mat1)):
            result = []
            for y in range(len(mat1[0])):
                result.append(mat1[x][y] + mat2[x][y])
            addMatrix.append(result)
        return addMatrix
    else:
        return None
