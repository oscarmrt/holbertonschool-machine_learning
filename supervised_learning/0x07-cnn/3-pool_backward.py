#!/usr/bin/env python3
"""Program that performs back propagation over a
pooling layer of a neural network"""
import numpy as np


def pool_backward(dA, A_prev, kernel_shape, stride=(1, 1), mode='max'):
    """Function that performs back propagation over a
    pooling layer of a neural network"""
    m, h_prev, w_prev, c_prev = A_prev.shape
    m, h_new, w_new, c_new = dA.shape
    kh, kw = kernel_shape
    sh, sw = stride
    dA_prev = np.zeros(A_prev.shape)
    for i in range(m):
        a_prev = A_prev[i]
        for j in range(h_new):
            for k in range(w_new):
                for p in range(c_new):
                    vert_start = j * sh
                    vert_end = vert_start + kh
                    horiz_start = k * sw
                    horiz_end = horiz_start + kw
                    if mode == "max":
                        a_prev_slice = a_prev[vert_start:vert_end,
                                              horiz_start:horiz_end, p]
                        mask = (a_prev_slice == np.max(a_prev_slice))
                        dA_prev[i, vert_start:vert_end,
                                horiz_start:horiz_end,
                                p] += np.multiply(mask, dA[i, j, k, p])
                    elif mode == "avg":
                        da = dA[i, j, k, p]
                        shape = kernel_shape
                        average = da / (kh * kw)
                        Z = np.ones(shape) * average
                        dA_prev[i, vert_start:vert_end,
                                horiz_start:horiz_end, p] += Z
    return dA_prev
