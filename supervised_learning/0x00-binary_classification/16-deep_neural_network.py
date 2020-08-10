#!/usr/bin/env python3
""" defines a deep neural network performing binary classification"""
import numpy as np


class DeepNeuralNetwork():
    """ defines a deep neural network performing binary classification
    using He initialization sqrt(2./layers_dims[l-1]).)"""

    def __init__(self, nx, layers):
        """Constructor for class DeepNeuralNetwork"""
        if type(nx) is not int:
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        if type(layers) is not list or not layers:
            raise TypeError("layers must be a list of positive integers")

        self.L = len(layers)
        self.cache = {}
        self.weights = {}

        for x in range(self.L):
            if type(layers[x]) is not int or layers[x] <= 0:
                raise TypeError('layers must be a list of positive integers')

            wi = 'W'+str(x + 1)
            bi = 'b'+str(x + 1)
            if x == 0:
                self.weights[wi] = np.random.randn(layers[x], nx)\
                                   * np.sqrt(2./nx)
            else:
                self.weights[wi] = np.random.randn(layers[x], layers[x-1])\
                                   * np.sqrt(2/layers[x-1])
            self.weights[bi] = np.zeros((layers[x], 1))
