#!/usr/bin/env python3
"""Program that performs forward propagation
over a pooling layer of a neural network"""
import numpy as np


def pool_forward(A_prev, kernel_shape, stride=(1, 1), mode='max'):
    """Function that performs forward propagation
    over a pooling layer of a neural network"""
    m = A_prev.shape[0]
    h = A_prev.shape[1]
    w = A_prev.shape[2]
    c = A_prev.shape[3]
    kh = kernel_shape[0]
    kw = kernel_shape[1]
    sh = stride[0]
    sw = stride[1]
    if mode == 'max':
        pooling = np.max
    else:
        pooling = np.average
    newh = int(((h - kh) / sh)) + 1
    neww = int(((w - kw) / sw)) + 1
    conv = np.zeros([m, newh, neww, c])
    for i in range(newh):
        for j in range(neww):
            x = i * sh
            y = j * sw
            conv[:, i, j, :] = pooling(A_prev[:, x:x+kh, y:y+kw, :],
                                       axis=(1, 2))
    return conv
