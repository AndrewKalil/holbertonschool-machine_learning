#!/usr/bin/env python3
""" Regulization """
import numpy as np


def dropout_gradient_descent(Y, weights, cache, alpha, keep_prob, L):
    """conducts forward propagation using Dropout

    Args:
        Y is a one-hot numpy.ndarray of shape (classes, m) that contains
         the correct labels for the data
            classes is the number of classes
            m is the number of data points
        weights is a dictionary of the weights and biases of the neural network
        cache is a dictionary of the outputs and dropout masks of each layer of
         the neural network
        alpha is the learning rate
        L the number of layers in the network
        keep_prob is the probability that a node will be kept
    """

    weights2 = weights.copy()
    m = Y.shape[1]

    for i in reversed(range(L)):
        n = i + 1
        if (n == L):
            dz = cache["A" + str(n)] - Y
            dw = (np.matmul(cache["A" + str(i)], dz.T) / m).T
        else:
            dz1 = np.matmul(weights2["W" + str(n + 1)].T, current_dz)
            dz2 = 1 - cache["A" + str(n)]**2
            dz = dz1 * dz2 * cache['D' + str(n)] / keep_prob
            dw = np.matmul(dz, cache["A" + str(i)].T) / m

        db = np.sum(dz, axis=1, keepdims=True) / m

        weights["W" + str(n)] -= (alpha * dw)
        weights["b" + str(n)] -= (alpha * db)

        current_dz = dz
