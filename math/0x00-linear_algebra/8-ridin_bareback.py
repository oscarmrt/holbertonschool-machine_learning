#!/usr/bin/env python3
"""Program that performs matrix multiplication"""


def mat_mul(mat1, mat2):
    """Function that performs matrix multiplication"""
    mat1m = len(mat1)
    mat1n = len(mat1[0])
    mat2m = len(mat2)
    mat2n = len(mat2[0])
    result = [[0, 0, 0, 0],
              [0, 0, 0, 0],
              [0, 0, 0, 0]]
    if mat1n == mat2m:
        for x in range(mat1m):
            for y in range(mat2n):
                for z in range(mat2m):
                    result[x][y] += mat1[x][z] * mat2[z][y]
        return result
    else:
        result = None
