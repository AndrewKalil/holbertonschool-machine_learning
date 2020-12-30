#!/usr/bin/env python3
""" Neuron """
import numpy as np


class Neuron:
    """Neuron Class
    """

    def __init__(self, nx):
        """Instantiation method

        Args:
                nx (int): number of input features to the neuron
        """
        if type(nx) != int:
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        else:
            self.__W = np.random.normal(0, 1, (1, nx))
            self.__b = 0
            self.__A = 0

    @property
    def W(self):
        """W which stand for weight is the weight of the connection

        Returns:
                Private instance W
        """
        return self.__W

    @property
    def b(self):
        """b which stand for bias is the bias of the neuron which allows
    determines the output

        Returns:
                Private instance b
        """
        return self.__b

    @property
    def A(self):
        """A which stand for Activated output is the known output of
    the neuron (expected)

        Returns:
                Private instance A
        """
        return self.__A
