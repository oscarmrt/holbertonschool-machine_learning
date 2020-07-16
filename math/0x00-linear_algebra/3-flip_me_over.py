#!/usr/bin/env python3
"""Program that transpose a 2D matrix"""


def matrix_transpose(matrix):
    """Function that returns the transpose of
    a 2D matrix"""

    result = [[matrix[y][x] for y in range(len(matrix))]
              for x in range(len(matrix[0]))]
    return result
