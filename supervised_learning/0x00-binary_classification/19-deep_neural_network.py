#!/usr/bin/env python3
""" defines a deep neural network performing binary classification"""
import numpy as np


class DeepNeuralNetwork():
    """ defines a deep neural network performing binary classification
    using He initialization sqrt(2./lyrs_dims[l-1]).)"""

    def __init__(self, nx, lyrs):
        """Constructor for class DeepNeuralNetwork"""
        if type(nx) is not int:
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        if type(lyrs) is not list or not lyrs:
            raise TypeError("lyrs must be a list of positive integers")

        self.__L = len(lyrs)
        self.__cache = {}
        self.__weights = {}

        for x in range(self.__L):
            if type(lyrs[x]) is not int or lyrs[x] <= 0:
                raise TypeError('lyrs must be a list of positive integers')

            wi = 'W'+str(x + 1)
            bi = 'b'+str(x + 1)
            if x == 0:
                self.__weights[wi] = np.random.randn(lyrs[x], nx)\
                                   * np.sqrt(2./nx)
            else:
                self.__weights[wi] = np.random.randn(lyrs[x], lyrs[x-1])\
                                   * np.sqrt(2/lyrs[x-1])
            self.__weights[bi] = np.zeros((lyrs[x], 1))

    @property
    def L(self):
        """ getter function for lyrs in neural network"""
        return self.__L

    @property
    def cache(self):
        """getter function for intermediate values of network"""
        return self.__cache

    @property
    def weights(self):
        """ getter function for Weights and biased of network"""
        return self.__weights

    def forward_prop(self, X):
        """Calculates the forward propagation of the neural network"""
        self.__cache['A0'] = X
        for lyr in range(1, self.__L + 1):
            xi = self.__cache['A'+str(lyr-1)]
            z = np.dot(self.__weights['W'+str(lyr)], xi) +\
                self.__weights['b'+str(lyr)]
            sigmoid = 1 / (1 + np.exp(-z))
            self.__cache['A'+str(lyr)] = sigmoid
        return sigmoid, self.__cache

    def cost(self, Y, A):
        """Calculates the cost of the model using logistic regression"""
        m = Y.shape[1]
        cost = - (1 / m) * np.sum(
            np.multiply(
                Y, np.log(A)) + np.multiply(
                1 - Y, np.log(1.0000001 - A)))
        return cost
