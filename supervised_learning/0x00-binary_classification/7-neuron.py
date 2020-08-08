#!/usr/bin/env python3
"""defines a single neuron performing binary classification"""
import numpy as np
import matplotlib.pyplot as plt


class Neuron():
    """defines a single neuron performing binary classification"""

    def __init__(self, nx):
        """Constructor for class Neuron"""
        if type(nx) is not int:
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")

        self.__W = np.random.randn(nx).reshape(1, nx)
        self.__b = 0
        self.__A = 0

    @property
    def W(self):
        """ getter function for Weight"""
        return self.__W

    @property
    def b(self):
        """ getter function for bias"""
        return self.__b

    @property
    def A(self):
        """getter function for Activate output"""
        return self.__A

    def forward_prop(self, X):
        """Calculates the forward propagation of the neuron"""
        Z = np.matmul(self.W, X) + self.b
        sig = 1 / (1 + np.exp(-Z))
        self.__A = sig
        return self.__A

    def cost(self, Y, A):
        """Calculates the cost of the model using logistic regression"""
        m = Y.shape[1]
        cost = - (1 / m) * np.sum(
            np.multiply(
                Y, np.log(A)) + np.multiply(
                1 - Y, np.log(1.0000001 - A)))
        return cost

    def evaluate(self, X, Y):
        """Evaluates the neuron’s predictions"""
        self.forward_prop(X)
        cost = self.cost(Y, self.__A)
        pred = np.where(self.__A >= 0.5, 1, 0)
        return pred, cost

    def gradient_descent(self, X, Y, A, alpha=0.05):
        """Calculates one pass of gradient descent on the neuron"""
        m = Y.shape[1]
        dz = A - Y
        dW = (1 / m) * np.matmul(X, dz.T)
        db = (1 / m) * np.sum(dz)
        self.__W -= 1 * (alpha * dW).T
        self.__b -= 1 * (alpha * db)

    def train(self, X, Y, iterations=5000, alpha=0.05,
              verbose=True, graph=True, step=100):
        """Trains the neuron by updating the private
        attributes __W, __b, and __A"""
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
            cost = self.cost(Y, self.__A)
            if verbose and x % step == 0:
                print('Cost after {} iterations: {}'.format(x, cost))
                csts.append(cost)
                stps.append(x)
            if x < iterations:
                self.gradient_descent(X, Y, self.__A, alpha)
        if graph is True:
            plt.plot(np.squeeze(stps), np.squeeze(csts))
            plt.ylabel("cost")
            plt.xlabel("iteration")
            plt.title("Training Cost")
            plt.show()
        return self.evaluate(X, Y)
