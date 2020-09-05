#!/usr/bin/env python3
"""Program that performs pooling on images"""
import numpy as np


def pool(images, kernel_shape, stride, mode='max'):
    """Function that performs pooling on images"""
    m = images.shape[0]
    h = images.shape[1]
    w = images.shape[2]
    c = images.shape[3]
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
            conv[:, i, j, :] = pooling(images[:, x:x+kh, y:y+kw, :],
                                       axis=(1, 2))
    return conv
