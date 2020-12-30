#!/usr/bin/env python3
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
