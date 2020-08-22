#!/usr/bin/env python3
"""Program that normalizes (standardizes) a matrix"""
import numpy as np


def normalize(X, m, s):
    """Function that normalizes (standardizes) a matrix"""
    return (X - m) / s
