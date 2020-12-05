#!/usr/bin/env python3
"""Program that Create the class RNNCell that represents
a cell of a simple RNN"""
import numpy as np


class RNNCell():
    """class RNNCell"""
    def __init__(self, i, h, o):
        """class constructor"""
        self.Wh = np.random.normal(size=(i + h, h))
        self.Wy = np.random.normal(size=(h, o))
        self.bh = np.zeros((1, h))
        self.by = np.zeros((1, o))\


    def softmax(self, x):
        """Function that performs softmax"""
        return np.exp(x) / np.exp(x).sum(axis=1, keepdims=True)


    def forward(self, h_prev, x_t):
        """Function that performs forward propagation for one time step"""
        h_next = np.tanh(np.dot(np.hstack((h_prev, x_t)), self.Wh) + self.bh)
        y = np.dot(h_next, self.Wy) + self.by
        return h_next, self.softmax(y)
