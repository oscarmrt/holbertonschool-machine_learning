#!/usr/bin/env python3
"""Program that calculates the F1 score of a confusion matrix"""
import numpy as np
sensitivity = __import__('1-sensitivity').sensitivity
precision = __import__('2-precision').precision


def f1_score(confusion):
    """Function that calculates the F1 score of a confusion matrix"""
    f1Score = 2 * (precision(confusion) * sensitivity(confusion)) /\
        (precision(confusion) + sensitivity(confusion))
    return f1Score
