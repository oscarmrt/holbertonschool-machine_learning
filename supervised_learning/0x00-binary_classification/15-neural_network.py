#!/usr/bin/env python3
"""defines a neural network with one hidden layer performing
binary classification"""
import numpy as np
import matplotlib.pyplot as plt


class NeuralNetwork():
    """defines a neural network with one hidden layer performing
    binary classification:"""

    def __init__(self, nx, nodes):
        """Constructor for class NeuralNetwork"""
        if type(nx) is not int:
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        if type(nodes) is not int:
            raise TypeError("nodes must be an integer")
        if nodes < 1:
            raise ValueError("nodes must be a positive integer")

        self.__W1 = np.random.randn(nx, nodes).reshape(nodes, nx)
        self.__b1 = np.zeros(nodes).reshape(nodes, 1)
        self.__A1 = 0
        self.__W2 = np.random.randn(nodes).reshape(1, nodes)
        self.__b2 = 0
        self.__A2 = 0

    @property
    def W1(self):
        """ getter function for Weight"""
        return self.__W1

    @property
    def b1(self):
        """ getter function for bias"""
        return self.__b1

    @property
    def A1(self):
        """getter function for Activate output"""
        return self.__A1

    @property
    def W2(self):
        """ getter function for Weight"""
        return self.__W2

    @property
    def b2(self):
        """ getter function for bias"""
        return self.__b2

    @property
    def A2(self):
        """getter function for Activate output"""
        return self.__A2

    def forward_prop(self, X):
        """Calculates the forward propagation of the neural network"""
        Z1 = np.matmul(self.W1, X) + self.b1
        self.__A1 = 1 / (1 + np.exp(-Z1))
        Z2 = np.matmul(self.W2, self.__A1) + self.b2
        self.__A2 = 1 / (1 + np.exp(-Z2))
        return self.__A1, self.__A2

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
        self.forward_prop(X)
        cost = self.cost(Y, self.__A2)
        pred = np.where(self.__A2 >= 0.5, 1, 0)
        return pred, cost

    def gradient_descent(self, X, Y, A1, A2, alpha=0.05):
        """Calculates one pass of gradient descent on the neural network"""
        m = Y.shape[1]
        dz2 = A2 - Y
        dw2 = (1 / m) * np.matmul(dz2, A1.T)
        db2 = (1 / m) * np.sum(dz2, axis=1, keepdims=True)
        dz1 = np.matmul(self.__W2.T, dz2) * A1 * (1 - A1)
        dW1 = (1 / m) * np.matmul(dz1, X.T)
        db1 = (1/m) * np.sum(dz1, axis=1, keepdims=True)
        self.__W2 = self.__W2 - (alpha * dw2)
        self.__b2 = self.__b2 - (alpha * db2)
        self.__W1 = self.__W1 - (alpha * dW1)
        self.__b1 = self.__b1 - (alpha * db1)

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
            cost = self.cost(Y, self.__A2)
            if verbose and x % step == 0:
                print('Cost after {} iterations: {}'.format(x, cost))
                csts.append(cost)
                stps.append(x)
            if x < iterations:
                self.gradient_descent(X, Y, self.__A1, self.__A2, alpha)
        if graph is True:
            plt.plot(np.squeeze(stps), np.squeeze(csts))
            plt.ylabel("cost")
            plt.xlabel("iteration")
            plt.title("Training Cost")
            plt.show()
        return self.evaluate(X, Y)