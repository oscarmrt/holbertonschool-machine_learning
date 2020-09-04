#!/usr/bin/env python3
"""Program that performs a same convolution on grayscale images"""
import numpy as np


def convolve_grayscale_same(images, kernel):
    """Function that performs a same convolution on grayscale images"""
    m = images.shape[0]
    h = images.shape[1]
    w = images.shape[2]
    kh = kernel.shape[0]
    kw = kernel.shape[1]
    padh = int(kh / 2)
    padw = int(kw / 2)
    pad = ((0, 0), (padh, padh), (padw, padw))
    conv = np.zeros([m, h, w])
    imagePad = np.pad(images, pad_width=pad, mode='constant')
    for i in range(h):
        for j in range(w):
            image = imagePad[:, i:i+kh, j:j+kw]
            conv[:, i, j] = np.multiply(image, kernel).sum(axis=(1, 2))
    return conv
