#!/usr/bin/env python3
"""Program that performs forward propagation for a deep RNN"""
import numpy as np


def deep_rnn(rnn_cells, X, h_0):
    """Function that performs forward propagation for a deep RNN"""
    T = X.shape[0]
    H = [[h for h in h_0]]
    Y = []
    for x in range(T):
        Htemp = []
        x_t = X[x]
        for i, j in enumerate(rnn_cells):
            h, y = j.forward(H[i][i], x_t)
            Htemp.append(h)
            x_t = h
        H.append(Htemp)
        Y.append(y)
    H = np.array(H)
    Y = np.array(Y)
    return (H, Y)
