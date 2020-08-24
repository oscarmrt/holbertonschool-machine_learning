#!/usr/bin/env python3
"""Program that creates a confusion matrix"""
import numpy as np


def create_confusion_matrix(labels, logits):
    """Function that creates a confusion matrix"""
    confusionMatrix = np.matmul(labels.T, logits)
    return confusionMatrix
