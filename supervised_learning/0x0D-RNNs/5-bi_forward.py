#!/usr/bin/env python3
"""Program Create the class BidirectionalCell that represents
a bidirectional cell of an RNN"""
import numpy as np


class BidirectionalCell():
    """class BidirectionalCell"""
    def __init__(self, i, h, o):
        """class constructor"""
        self.Whf = np.random.normal(size=(i + h, h))
        self.Whb = np.random.normal(size=(i + h, h))
        self.Wy = np.random.normal(size=(h * 2, o))
        self.bhf = np.zeros((1, h))
        self.bhb = np.zeros((1, h))
        self.by = np.zeros((1, o))

    def forward(self, h_prev, x_t):
        """Function that calculates the hidden state in the forward
        direction for one time step"""
        h_next = np.tanh(np.dot(np.hstack((h_prev, x_t)), self.Whf) + self.bhf)
        return h_next
