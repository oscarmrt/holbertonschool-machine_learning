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

        self.__L = len(layers)
        self.__cache = {}
        self.__weights = {}

        for x in range(self.__L):
            if type(layers[x]) is not int or layers[x] <= 0:
                raise TypeError('layers must be a list of positive integers')

            wi = 'W'+str(x + 1)
            bi = 'b'+str(x + 1)
            if x == 0:
                self.__weights[wi] = np.random.randn(layers[x], nx)\
                                   * np.sqrt(2./nx)
            else:
                self.__weights[wi] = np.random.randn(layers[x], layers[x-1])\
                                   * np.sqrt(2/layers[x-1])
            self.__weights[bi] = np.zeros((layers[x], 1))

    @property
    def L(self):
        """ getter function for layers in neural network"""
        return self.__L

    @property
    def cache(self):
        """getter function for intermediate values of network"""
        return self.__cache

    @property
    def weights(self):
        """ getter function for Weights and biased of network"""
        return self.__weights
