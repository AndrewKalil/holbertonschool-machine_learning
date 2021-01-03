#!/usr/bin/env python3
""" Deep Neural Network """
import numpy as np


class DeepNeuralNetwork:
    """defines a deep neural network performing binary classification
    """

    def __init__(self, nx, layers):
        """Instantiation Method

        Args:
            nx: number of input features
            layers: list representing the number of nodes in each
                layer of the network
        """
        if type(nx) != int:
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        if type(layers) != list:
            raise TypeError("layers must be a list of positive integers")
        if len(layers) == 0:
            raise TypeError("layers must be a list of positive integers")
        self.nx = nx
        self.layers = layers
        self.__L = len(layers)
        self.__cache = {}
        self.__weights = {}

        for i in range(self.L):
            if type(layers[i]) != int or layers[i] < 1:
                raise TypeError("layers must be a list of positive integers")
            W_key = "W{}".format(i + 1)
            b_key = "b{}".format(i + 1)

            self.weights[b_key] = np.zeros((layers[i], 1))

            if i == 0:
                f = np.sqrt(2 / nx)
                self.weights['W1'] = np.random.randn(layers[i], nx) * f
            else:
                f = np.sqrt(2 / layers[i - 1])
                self.weights[W_key] = np.random.randn(layers[i],
                                                      layers[i - 1]) * f

    @property
    def L(self):
        """ property setter for the attribute """
        return self.__L

    @property
    def cache(self):
        """ property setter for the attribute """
        return self.__cache

    @property
    def weights(self):
        """ property setter for the attribute """
        return self.__weights

    def forward_prop(self, X):
        """Calculates the forward propagation of the
            neural network

        Args:
            X: numpy.ndarray with shape (nx, m) that contains the input data
                nx is the number of input features to the neuron
                m is the number of examples
        """
        self.__cache['A0'] = X

        for i in range(self.__L):
            W_key = "W{}".format(i + 1)
            b_key = "b{}".format(i + 1)
            A_key_prev = "A{}".format(i)
            A_key_forw = "A{}".format(i + 1)

            Z = np.matmul(self.__weights[W_key], self.__cache[A_key_prev]) \
                + self.__weights[b_key]
            self.__cache[A_key_forw] = 1 / (1 + np.exp(-Z))

        return self.__cache[A_key_forw], self.__cache

    def cost(self, Y, A):
        """Calculates the cost of the model using logistic regression

        Args:
            Y: numpy.ndarray with shape (1, m) that contains the
                correct labels for the input data
            A: numpy.ndarray with shape (1, m) containing the activated
                output of the neuron for each example
        """
        cost = -np.sum((Y * np.log(A)) +
                       ((1 - Y) * np.log(1.0000001 - A))) / Y.shape[1]
        return cost
