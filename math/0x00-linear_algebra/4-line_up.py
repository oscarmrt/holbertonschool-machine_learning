#!/usr/bin/env python3
"""program that adds two arrays element-wise"""


def add_arrays(arr1, arr2):
    """Function that adds two arrays element-wise"""

    if len(arr1) != len(arr2):
        return None
    newList = []
    for x, y in zip(arr1, arr2):
        adt = x + y
        newList.append(adt)
    return newList
