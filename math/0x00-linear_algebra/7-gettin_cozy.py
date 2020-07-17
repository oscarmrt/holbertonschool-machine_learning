#!/usr/bin/env python3
"""Program that concatenates two matrices
along a specific axis"""


def cat_matrices2D(mat1, mat2, axis=0):
    """Function that concatenates two matrices
    along a specific axis"""
    x = [row[:] for row in mat1]
    y = [row[:] for row in mat2]
    mat1m = len(mat1)
    mat1n = len(mat1[0])
    mat2m = len(mat2)
    mat2n = len(mat2[0])
    newArray = []
    if axis == 0:
        if mat1n != mat2n:
            return None
        return x + y
    if axis == 1:
        if mat1m != mat2m:
            return None
        for x, y in zip(mat1, mat2):
            adt = x + y
            newArray.append(adt)
        return newArray
