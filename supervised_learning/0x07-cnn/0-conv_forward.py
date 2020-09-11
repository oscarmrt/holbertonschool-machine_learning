#!/usr/bin/env python3
"""Program that performs forward propagation over
a convolutional layer of a neural network"""
import numpy as np


def conv_forward(A_prev, W, b, activation, padding="same", stride=(1, 1)):
    """Function that performs forward propagation over
    a convolutional layer of a neural network"""
    m, h_prev, w_prev, c_prev = A_prev.shape
    kh, kw, c_prev, c_new = W.shape
    sh, sw = stride
    if padding == 'same':
        padh = int((((h_prev - 1) * sh - h_prev + kh) / 2))
        padw = int((((w_prev - 1) * sw - w_prev + kw) / 2))
    else:
        padh = 0
        padw = 0
    pad = ((0, 0), (padh, padh), (padw, padw), (0, 0))
    newh = int(((h_prev + (2 * padh) - kh) / sh)) + 1
    neww = int(((w_prev + (2 * padw) - kw) / sw)) + 1
    conv = np.zeros([m, newh, neww, c_new])
    imagePad = np.pad(A_prev, pad_width=pad, mode='constant')
    for i in range(newh):
        for j in range(neww):
            for k in range(c_new):
                x = i * sh
                y = j * sw
                image = imagePad[:, x:x+kh, y:y+kw, :]
                conv[:, i, j, k] = np.multiply(image, W[:, :, :, k]).\
                    sum(axis=(1, 2, 3))
    Z = conv + b
    return activation(Z)
