#!/usr/bin/env python3
"""Program that calculates the specificity for
each class in a confusion matrix"""
import numpy as np


def specificity(confusion):
    """Function that calculates the specificity for
    each class in a confusion matrix"""
    TP = np.diag(confusion)
    FP = confusion.sum(axis=0) - np.diag(confusion)
    FN = confusion.sum(axis=1) - np.diag(confusion)
    TN = np.sum(confusion) - (FP + FN + TP)
    TNR = TN/(TN+FP)
    return TNR
