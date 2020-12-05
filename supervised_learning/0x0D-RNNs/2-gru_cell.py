#!/usr/bin/env python3
"""Program that Create the class GRUCell that represents
a gated recurrent unit"""
import numpy as np


class GRUCell():
    """class GRUCell"""
    def __init__(self, i, h, o):
        """class constructor"""
        self.Wz = np.random.normal(size=(i + h, h))
        self.Wr = np.random.normal(size=(i + h, h))
        self.Wh = np.random.normal(size=(i + h, h))
        self.Wy = np.random.normal(size=(h, o))
        self.bz = np.zeros((1, h))
        self.br = np.zeros((1, h))
        self.bh = np.zeros((1, h))
        self.by = np.zeros((1, o))

    def softmax(self, x):
        """Function that performs softmax"""
        return np.exp(x) / np.exp(x).sum(axis=1, keepdims=True)

    def sigmoid(self, x):
        """Function that performs Sigmoid"""
        return (1 / (1 + np.exp(-x)))

    def forward(self, h_prev, x_t):
        """Function that performs forward propagation for one time step"""
        U = np.hstack((h_prev, x_t))
        z = self.sigmoid(np.dot(U, self.Wz) + self.bz)
        r = self.sigmoid(np.dot(U, self.Wr) + self.br)
        U = np.hstack((h_prev * r, x_t))
        c = np.tanh(np.dot(U, self.Wh) + self.bh)
        h_next = np.multiply(z, c) + np.multiply((1 - z), h_prev)
        y = self.softmax(np.dot(h_next, self.Wy) + self.by)
        return (h_next, y)
