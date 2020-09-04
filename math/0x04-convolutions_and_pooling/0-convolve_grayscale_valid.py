#!/usr/bin/env python3
"""Program that performs a valid convolution on grayscale images"""
import numpy as np


def convolve_grayscale_valid(images, kernel):
    """Function that performs a valid convolution on grayscale images"""
    m = images.shape[0]
    h = images.shape[1]
    w = images.shape[2]
    kh = kernel.shape[0]
    kw = kernel.shape[1]
    hConv = h - kh + 1
    wConv = w - kw + 1
    conv = np.zeros([m, hConv, wConv])
    for i in range(hConv):
        for j in range(wConv):
            image = images[:, i:i+kh, j:j+kw]
            conv[:, i, j] = np.multiply(image, kernel).sum(axis=(1, 2))
    return conv
