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

    def backward(self, h_next, x_t):
        """that calculates the hidden state in the backward
        direction for one time step"""
        h_prev = np.tanh(np.dot(np.hstack((h_next, x_t)), self.Whb) + self.bhb)
        return h_prev

    def softmax(self, x):
        """Function that performs softmax"""
        return np.exp(x) / np.exp(x).sum(axis=1, keepdims=True)

    def output(self, H):
        """Function  that calculates all outputs for the RNN"""
        T = H.shape[0]
        Y = []
        for i in range(T):
            Y.append(self.softmax(np.dot(H[i], self.Wy) + self.by))
        return np.array(z)
