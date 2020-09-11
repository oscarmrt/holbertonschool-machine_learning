#!/usr/bin/env python3
"""Program that performs back propagation over a
convolutional layer of a neural network"""
import numpy as np


def conv_backward(dZ, A_prev, W, b, padding="same", stride=(1, 1)):
    """Function that performs back propagation over a
    convolutional layer of a neural network:"""
    m, h_prev, w_prev, c_prev = A_prev.shape
    kh, kw, c_prev, c_new = W.shape
    sh, sw = stride
    (m, h_new, w_new, c_new) = dZ.shape
    if padding == 'same':
        padh = int(np.ceil((((h_prev - 1) * sh - h_prev + kh) / 2)))
        padw = int(np.ceil((((w_prev - 1) * sw - w_prev + kw) / 2)))
    elif padding == 'valid':
        padh = 0
        padw = 0
    dA_prev = np.zeros(A_prev.shape)
    dW = np.zeros(W.shape)
    db = np.sum(dZ, axis=(0, 1, 2), keepdims=True)
    pad = ((0, 0), (padh, padh), (padw, padw), (0, 0))
    A_prev_pad = np.pad(A_prev, pad_width=pad, mode='constant')
    dA_prev_pad = np.pad(dA_prev, pad_width=pad, mode='constant')
    for i in range(m):
        a_prev_pad = A_prev_pad[i]
        da_prev_pad = dA_prev_pad[i]
        for j in range(h_new):
            for k in range(w_new):
                for p in range(c_new):
                    vert_start = j * sh
                    vert_end = vert_start + kh
                    horiz_start = k * sw
                    horiz_end = horiz_start + kw
                    a_slice = a_prev_pad[vert_start:vert_end,
                                         horiz_start:horiz_end]
                    da_prev_pad[vert_start:vert_end,
                                horiz_start:horiz_end] += W[:, :, :, p] *\
                        dZ[i, j, k, p]
                    dW[:, :, :, p] += a_slice * dZ[i, j, k, p]
        if padding == 'same':
            dA_prev[i, :, :, :] += da_prev_pad[padh:-padh, padw:-padw, :]
        if padding == 'valid':
            dA_prev[i, :, :, :] += da_prev_pad
    return dA_prev, dW, db
