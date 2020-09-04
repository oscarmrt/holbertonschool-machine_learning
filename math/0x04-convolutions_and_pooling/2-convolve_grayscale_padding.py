#!/usr/bin/env python3
"""Program that performs a convolution on
grayscale images with custom padding"""
import numpy as np


def convolve_grayscale_padding(images, kernel, padding):
    """Function that performs a convolution on
    grayscale images with custom padding"""
    m = images.shape[0]
    h = images.shape[1]
    w = images.shape[2]
    kh = kernel.shape[0]
    kw = kernel.shape[1]
    padh = padding[0]
    padw = padding[1]
    pad = ((0, 0), (padh, padh), (padw, padw))
    newh = h + (2 * padh) - kh + 1
    neww = w + (2 * padw) - kw + 1
    conv = np.zeros([m, newh, neww])
    imagePad = np.pad(images, pad_width=pad, mode='constant')
    for i in range(newh):
        for j in range(neww):
            image = imagePad[:, i:i+kh, j:j+kw]
            conv[:, i, j] = np.multiply(image, kernel).sum(axis=(1, 2))
    return conv
