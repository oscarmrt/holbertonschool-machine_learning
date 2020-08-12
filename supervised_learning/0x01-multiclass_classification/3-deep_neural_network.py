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
        self.__cache["A0"] = X
        for i in range(self.__L):
            weights = self.__weights
            cache = self.__cache
            w_i = "W" + str(i + 1)
            b_i = "b" + str(i + 1)
            Za = np.matmul(weights[w_i], cache["A" + str(i)])
            Z = Za + weights[b_i]
            if i == self.__L - 1:
                t = np.exp(Z)
                cache["A" + str(i + 1)] = (t/np.sum(t, axis=0, keepdims=True))
            else:
                cache["A" + str(i + 1)] = 1 / (1 + np.exp(-Z))
        return cache["A" + str(self.__L)], cache

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
