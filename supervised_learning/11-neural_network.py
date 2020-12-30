#!/usr/bin/python3
""" Neural Network """
import numpy as np


class NeuralNetwork:
    """that defines a neural network with one hidden layer performing
    binary classification"""

    def __init__(self, nx, nodes):
        """Instantiation method

        Args:
            nx: the number of input features
            nodes: the number of nodes found in the hidden layer
        """
        if type(nx) != int:
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        if type(nodes) != int:
            raise TypeError("nodes must be an integer")
        if nodes < 1:
            raise ValueError("nodes must be a positive integer")
        # The weights vector for the hidden layer
        self.__W1 = np.random.normal(0, 1, (nodes, nx))
        # The bias for the hidden layer
        self.__b1 = np.zeros((nodes, 1))
        # The activated output for the hidden layer
        self.__A1 = 0
        # The weights vector for the output neuron
        self.__W2 = np.random.normal(0, 1, (1, nodes))
        # The bias for the output neuron
        self.__b2 = 0
        # The activated output for the output neuron (prediction)
        self.__A2 = 0

    @property
    def W1(self):
        """ Private instance retriever """
        return self.__W1

    @property
    def b1(self):
        """ Private instance retriever """
        return self.__b1

    @property
    def A1(self):
        """ Private instance retriever """
        return self.__A1

    @property
    def W2(self):
        """ Private instance retriever """
        return self.__W2

    @property
    def b2(self):
        """ Private instance retriever """
        return self.__b2

    @property
    def A2(self):
        """ Private instance retriever """
        return self.__A2

    def forward_prop(self, X):
        """defines a neural network with one hidden layer
            performing binary classification

        Args:
            X: numpy.ndarray with shape (nx, m) that contains the input data
                nx is the number of input features to the neuron
                m is the number of examples
        """
        Z1 = np.matmul(self.__W1, X) + self.__b1
        self.__A1 = 1 / (1 + np.exp(-Z1))
        Z2 = np.matmul(self.__W2, self.__A1) + self.__b2
        self.__A2 = 1 / (1 + np.exp(-Z2))
        return self.__A1, self.__A2

    def cost(self, Y, A):
        """Calculates the cost of the model using logistic regression

        Args:
            Y: numpy.ndarray with shape (1, m) that contains the
                correct labels for the input data
            A: numpy.ndarray with shape (1, m) containing the activated
                output of the neuron for each
        """
        cost = -np.sum((Y * np.log(A)) +
                       ((1 - Y) * np.log(1.0000001 - A))) / Y.shape[1]
        return cost
