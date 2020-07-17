#!/usr/bin/env python3
"""Program that performs matrix multiplication"""


def mat_mul(mat1, mat2):
    """Function that performs matrix multiplication"""
    mat1m = len(mat1)
    mat1n = len(mat1[0])
    mat2m = len(mat2)
    mat2n = len(mat2[0])
    result = []
    if mat1n == mat2m:
        for x in range(mat1m):
            line = [0] * mat2n
            result.append(line)
        for y in range(len(result)):
            for z in range(len(result[0])):
                for i in range(mat1n):
                    result[y][z] += mat1[y][i] * mat2[i][z]
        return result
    return None
