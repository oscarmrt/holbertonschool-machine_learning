#!/usr/bin/env python3
""" Creates a deep neural network. """
import numpy as np
import matplotlib.pyplot as plt
import pickle


class DeepNeuralNetwork:
    """ Deep neural network class. """

    def __init__(self, nx, layers):
        """ Initializer for the deep neural network. """
        if type(nx) != int:
            raise TypeError('nx must be an integer')
        if nx < 1:
            raise ValueError('nx must be a positive integer')
        if type(layers) != list or not layers:
            raise TypeError('layers must be a list of positive integers')
        self.__L = 0
        self.__cache = {}
        self.__weights = {}
        rand = np.random.randn
        for idx, neurons in enumerate(layers):
            if type(neurons) != int or neurons <= 0:
                raise TypeError('layers must be a list of positive integers')
            if idx == 0:
                self.weights['W1'] = rand(neurons, nx) * np.sqrt(2 / nx)
            else:
                p = layers[idx - 1]
                r = rand(neurons, p)
                self.weights["W{}".format(idx + 1)] = r * np.sqrt(2 / p)
            self.__L += 1
            self.weights["b{}".format(idx + 1)] = np.zeros((neurons, 1))

    @property
    def L(self):
        """ Getter for L (Number of layers). """
        return self.__L

    @property
    def cache(self):
        """ Getter for cache. """
        return self.__cache

    @property
    def weights(self):
        """ Getter for weights. """
        return self.__weights

    def forward_prop(self, X):
        """ Forward propagation of the network. """
        self.__cache['A0'] = X
        for i in range(self.L):
            w = self.weights['W{}'.format(i + 1)]
            b = self.weights['b{}'.format(i + 1)]
            p_a = self.cache['A' + str(i)]
            A = 1 / (1 + np.exp(-(np.matmul(w, p_a) + b)))
            self.__cache["A" + str(i + 1)] = A
        return (self.__cache["A" + str(i + 1)], self.cache)

    def cost(self, Y, A):
        """ Calculates the cost of the network. """
        m = Y.shape[1]
        c = np.sum(-Y * np.log(A) - (1 - Y) * np.log(1.0000001 - A)) / m
        return c

    def evaluate(self, X, Y):
        """ Evaluates the output of the network. """
        A, _ = self.forward_prop(X)
        c = self.cost(Y, A)
        return (np.where(A >= 0.5, 1, 0), c)

    def gradient_descent(self, Y, cache, alpha=0.05):
        """ Calculates the gradient descent. """
        m = Y.shape[1]
        oldW = self.weights.copy()
        for i in range(self.L, 0, -1):
            A = cache["A" + str(i)]
            if i == self.L:
                dz = A - Y
            else:
                dz = np.matmul(oldW["W" + str(i + 1)].T, dz) * A * (1 - A)
            dw = np.matmul(dz, cache["A" + str(i - 1)].T) / m
            db = np.sum(dz, axis=1, keepdims=True) / m
            w = oldW["W" + str(i)]
            b = oldW["b" + str(i)]
            self.__weights["W" + str(i)] = w - alpha * dw
            self.__weights["b" + str(i)] = b - alpha * db

    def train(self, X, Y, iterations=5000, alpha=0.05, verbose=True,
              graph=True, step=100):
        """ Trains the network. """
        if type(iterations) != int:
            raise TypeError('iterations must be an integer')
        if iterations < 1:
            raise ValueError('iterations must be a positive integer')
        if type(alpha) != float:
            raise TypeError('alpha must be a float')
        if alpha <= 0:
            raise ValueError('alpha must be positive')
        if verbose or graph:
            if type(step) != int:
                raise TypeError('step must be an integer')
            if step < 1 or step > iterations:
                raise ValueError('step must be positive and <= iterations')
        iteration = 0
        x_data = np.arange(0, iterations + 1, step)
        y_data = []
        for i in range(iterations):
            A, c = self.evaluate(X, Y)
            if iteration % step == 0:
                if graph:
                    y_data.append(c)
                if verbose:
                    print('Cost after {} iterations: {}'.format(iteration, c))
            iteration += 1
            self.gradient_descent(Y, self.cache, alpha)
        A, c = self.evaluate(X, Y)
        if verbose:
            print('Cost after {} iterations: {}'.format(iteration, c))
        if graph:
            y_data.append(c)
            plt.plot(x_data, y_data, 'b')
            plt.xlabel('iteration')
            plt.ylabel('cost')
            plt.title('Training Cost')
            plt.show()
        return (A, c)

    def save(self, filename):
        """ Saves the instance object to a file in pickle format. """
        if not filename.endswith(".pkl"):
            filename += ".pkl"
        with open(filename, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def load(filename):
        """ Loads a pickled DeepNeuralNetwork object. """
        try:
            with open(filename, 'rb') as f:
                loaded = pickle.load(f)
                return loaded
        except FileNotFoundError:
            return None
