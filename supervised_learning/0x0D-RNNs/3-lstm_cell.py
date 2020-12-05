#!/usr/bin/env python3
"""Program Create the class LSTMCell that represents an LSTM unit"""
import numpy as np


class LSTMCell():
    """class LSTMCell"""
    def __init__(self, i, h, o):
        """class constructor"""
        self.Wf = np.random.normal(size=(i + h, h))
        self.Wu = np.random.normal(size=(i + h, h))
        self.Wc = np.random.normal(size=(i + h, h))
        self.Wo = np.random.normal(size=(i + h, h))
        self.Wy = np.random.normal(size=(h, o))
        self.bf = np.zeros((1, h))
        self.bu = np.zeros((1, h))
        self.bc = np.zeros((1, h))
        self.bo = np.zeros((1, h))
        self.by = np.zeros((1, o))

    def softmax(self, x):
        """Function that performs softmax"""
        return np.exp(x) / np.exp(x).sum(axis=1, keepdims=True)

    def sigmoid(self, x):
        """Function that performs Sigmoid"""
        return (1 / (1 + np.exp(-x)))

    def forward(self, h_prev, c_prev, x_t):
        """Function that performs forward propagation for one time step"""
        U = np.hstack((h_prev, x_t))
        f = self.sigmoid(np.dot(U, self.Wf) + self.bf)
        u = self.sigmoid(np.dot(U, self.Wu) + self.bu)
        c_bar = np.tanh(np.dot(U, self.Wc) + self.bc)
        c_next = f * c_prev + u * c_bar
        o = self.sigmoid(np.dot(U, self.Wo) + self.bo)
        h_next = o * np.tanh(c_next)
        y = self.softmax(np.dot(h_next, self.Wy) + self.by)
        return (h_next, c_next, y)
