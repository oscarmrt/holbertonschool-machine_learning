#!/usr/bin/env python3
"""Program that calculates the sensitivity
for each class in a confusion matrix"""
import numpy as np


def sensitivity(confusion):
    """Function that calculates the sensitivity
    for each class in a confusion matrix"""
    TP = np.diag(confusion)
    FN = confusion.sum(axis=1) - np.diag(confusion)
    TPR = TP/(TP+FN)
    return TPR
