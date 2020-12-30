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
        self.W1 = np.random.normal(0, 1, (nodes, nx))
        # The bias for the hidden layer
        self.b1 = np.zeros((nodes, 1))
        # The activated output for the hidden layer
        self.A1 = 0
        # The weights vector for the output neuron
        self.W2 = np.random.normal(0, 1, (1, nodes))
        # The bias for the output neuron
        self.b2 = 0
        # The activated output for the output neuron (prediction)
        self.A2 = 0
