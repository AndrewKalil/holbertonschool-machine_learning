#!/usr/bin/env python3
""" Regulization """
import numpy as np


def l2_reg_cost(cost, lambtha, weights, L, m):
    """calculates the cost of a neural network with L2 regularization

    Args:
        cost is the cost of the network without L2 regularization
        lambtha is the regularization parameter
        weights is a dictionary of the weights and biases
         (numpy.ndarrays) of the neural network
        L is the number of layers in the neural network
        m is the number of data points used
    """
    summation = 0
    for ly in range(1, L + 1):
        key = "W{}".format(ly)
        summation += np.linalg.norm(weights[key])

    L2_cost = lambtha * summation / (2 * m)

    return cost + L2_cost