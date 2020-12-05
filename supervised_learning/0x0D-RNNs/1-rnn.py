#!/usr/bin/env python3
"""Program that performs forward propagation for a simple RNN"""
import numpy as np


def rnn(rnn_cell, X, h_0):
    """Function  that performs forward propagation for a simple RNN"""
    T = X.shape[0]
    H = []
    Y = []
    h = h_0
    H.append(h)
    for i in range(T):
        h, y = rnn_cell.forward(h, X[i])
        H.append(h)
        Y.append(y)
    H = np.array(H)
    Y = np.array(Y)
    return (H, Y)
