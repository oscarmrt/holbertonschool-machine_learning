#!/usr/bin/env python3
"""Program that performs a convolution on grayscale images:"""
import numpy as np


def convolve_grayscale(images, kernel, padding='same', stride=(1, 1)):
    """Function that performs a convolution on grayscale images:"""
    m = images.shape[0]
    h = images.shape[1]
    w = images.shape[2]
    kh = kernel.shape[0]
    kw = kernel.shape[1]
    sh = stride[0]
    sw = stride[1]
    if type(padding) is tuple:
        padh = padding[0]
        padw = padding[1]
    elif padding == 'same':
        padh = int((((h - 1) * sh - h + kh) / 2)) + 1
        padw = int((((w - 1) * sw - w + kw) / 2)) + 1
    elif padding == 'valid':
        padh = 0
        padw = 0
    pad = ((0, 0), (padh, padh), (padw, padw))
    newh = int(((h + (2 * padh) - kh) / sh)) + 1
    neww = int(((w + (2 * padw) - kw) / sw)) + 1
    conv = np.zeros([m, newh, neww])
    imagePad = np.pad(images, pad_width=pad, mode='constant')
    for i in range(newh):
        for j in range(neww):
            x = i * sh
            y = j * sw
            image = imagePad[:, x:x+kh, y:y+kw]
            conv[:, i, j] = np.multiply(image, kernel).sum(axis=(1, 2))
    return conv
