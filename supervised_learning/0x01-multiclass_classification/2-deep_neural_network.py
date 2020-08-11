#!/usr/bin/env python3
""" defines a deep neural network performing binary classification"""
import numpy as np
import matplotlib.pyplot as plt
import pickle


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

    def evaluate(self, X, Y):
        """Evaluates the neural network’s predictions"""
        prediction, cache = self.forward_prop(X)
        cost = self.cost(Y, prediction)
        return np.where(prediction >= 0.5, 1, 0), cost

    def gradient_descent(self, Y, cache, alpha=0.05):
        """Calculates one pass of gradient descent
        on the neural network"""
        m = Y.shape[1]
        dz = cache['A'+str(self.__L)] - Y
        for i in range(self.__L, 0, -1):
            db = (1 / m) * np.sum(dz, axis=1, keepdims=True)
            dW = (1 / m) * np.matmul(cache['A'+str(i-1)], dz.T)
            dz = np.matmul(self.__weights['W'+str(i)].T, dz) *\
                (cache['A'+str(i-1)] * (1 - cache['A'+str(i-1)]))
            self.__weights['W'+str(i)] = self.__weights['W'+str(i)] -\
                (alpha * dW).T
            self.__weights['b'+str(i)] = self.__weights['b'+str(i)] -\
                (alpha * db)

    def train(self, X, Y, iterations=5000, alpha=0.05,
              verbose=True, graph=True, step=100):
        """Trains the neural network"""
        if type(iterations) is not int:
            raise TypeError("iterations must be an integer")
        if iterations < 0:
            raise ValueError("iterations must be a positive integer")
        if type(alpha) is not float:
            raise TypeError("alpha must be a float")
        if alpha < 0:
            raise ValueError("alpha must be positive")
        if type(step) is not int:
            raise TypeError("step must be an integer")
        if step < 0 or step > iterations:
            raise ValueError("step must be positive and <= iterations")
        csts = []
        stps = []
        for x in range(iterations + 1):
            self.forward_prop(X)
            aux, cost = self.evaluate(X, Y)
            if verbose and x % step == 0:
                print('Cost after {} iterations: {}'.format(x, cost))
                csts.append(cost)
                stps.append(x)
            if x < iterations:
                self.gradient_descent(Y, self.__cache, alpha)
        if graph is True:
            plt.plot(np.squeeze(stps), np.squeeze(csts))
            plt.ylabel("cost")
            plt.xlabel("iteration")
            plt.title("Training Cost")
            plt.show()
        return self.evaluate(X, Y)

    def save(self, filename):
        """Saves the instance object to a file in pickle format"""
        if '.pkl' not in filename:
            filename += '.pkl'
        fileObject = open(filename, 'wb')
        pickle.dump(self, fileObject)
        fileObject.close()

    @staticmethod
    def load(filename):
        """Loads a pickled DeepNeuralNetwork object"""
        try:
            with open(filename, 'rb') as fileObject:
                fileOpen = pickle.load(fileObject)
            return fileOpen
        except FileNotFoundError:
            return None
