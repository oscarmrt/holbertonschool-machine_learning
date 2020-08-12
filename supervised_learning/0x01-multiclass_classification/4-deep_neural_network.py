#!/usr/bin/env python3
""" defines a deep neural network performing binary classification"""
import numpy as np
import matplotlib.pyplot as plt
import pickle


class DeepNeuralNetwork():
    """ defines a deep neural network performing binary classification
    using He initialization sqrt(2./lyrs_dims[l-1]).)"""

    def __init__(self, nx, layers, activation='sig'):
        """Constructor for class DeepNeuralNetwork"""
        if type(nx) != int:
            raise TypeError('nx must be an integer')
        if nx < 1:
            raise ValueError('nx must be a positive integer')
        if type(layers) != list or not layers:
            raise TypeError('layers must be a list of positive integers')
        if activation not in {'sig', 'tanh'}:
            raise ValueError("activation must be 'sig' or 'tanh'")
        self.__L = 0
        self.__cache = {}
        self.__weights = {}
        rand = np.random.randn
        self.__activation = activation
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

    @property
    def activation(self):
        """ Getter for the activation function used. """
        return self.__activation

    def forward_prop(self, X):
        """Calculates the forward propagation of the neural network"""
        self.__cache['A0'] = X
        for i in range(self.L):
            w = self.weights['W{}'.format(i + 1)]
            b = self.weights['b{}'.format(i + 1)]
            p_a = self.cache['A' + str(i)]
            out = np.matmul(w, p_a) + b
            if i == self.L - 1:
                denominator = np.sum(np.exp(out), axis=0)
                self.__cache["A" + str(i + 1)] = np.exp(out) / denominator
            else:
                if self.activation == 'sig':
                    self.__cache["A" + str(i + 1)] = 1 / (1 + np.exp(-out))
                else:
                    self.__cache["A" + str(i + 1)] = np.tanh(out)
        return (self.__cache["A" + str(i + 1)], self.cache)

    def cost(self, Y, A):
        """Calculates the cost of the model using logistic regression"""
        m = Y.shape[1]
        cost = - (1 / m) * np.sum(Y * np.log(A))
        return cost

    def evaluate(self, X, Y):
        """Evaluates the neural network’s predictions"""
        A, cache = self.forward_prop(X)
        cost = self.cost(Y, A)
        prediction = np.where(A >= 0.5, 1, 0)
        return prediction, cost

    def gradient_descent(self, Y, cache, alpha=0.05):
        """Calculates one pass of gradient descent
        on the neural network"""
        m = Y.shape[1]
        cpW = self.weights.copy()
        for i in range(self.L, 0, -1):
            A = cache["A" + str(i)]
            if i == self.L:
                dz = A - Y
            else:
                if self.activation == 'sig':
                    dz = np.matmul(cpW["W" + str(i + 1)].T, dz) * A * (1 - A)
                else:
                    dz = np.matmul(cpW["W" + str(i + 1)].T, dz) * (1 - A ** 2)
            dw = np.matmul(dz, cache["A" + str(i - 1)].T) / m
            db = np.sum(dz, axis=1, keepdims=True) / m
            w = cpW["W" + str(i)]
            b = cpW["b" + str(i)]
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
