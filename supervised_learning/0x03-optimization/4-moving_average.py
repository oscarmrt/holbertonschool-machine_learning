#!/usr/bin/env python3
"""Program that calculates the weighted moving average of a data set"""

import numpy as np


def moving_average(data, beta):
    """Function that calculates the weighted moving average of a data set"""
    vt = 0
    mAverage = []
    for x in range(len(data)):
        vt = (beta * vt) + ((1 - beta) * data[x])
        newBias = 1 - beta ** (x + 1)
        newVt = vt / newBias
        mAverage.append(newVt)
    return mAverage
