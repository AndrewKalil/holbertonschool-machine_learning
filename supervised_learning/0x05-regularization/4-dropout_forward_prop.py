#!/usr/bin/env python3
""" Regulization """
import numpy as np


def dropout_forward_prop(X, weights, L, keep_prob):
    """conducts forward propagation using Dropout

    Args:
        X is a numpy.ndarray of shape (nx, m) containing the input data for
         the network
            nx is the number of input features
            m is the number of data points
        weights is a dictionary of the weights and biases of the neural network
        L the number of layers in the network
        keep_prob is the probability that a node will be kept
    """

    cache = {}
    cache['A0'] = X

    for la in range(1, L + 1):
        keyA = "A{}".format(la)
        keyA_p = "A{}".format(la - 1)
        keyD = "D{}".format(la)
        keyW = "W{}".format(la)
        keyb = "b{}".format(la)

        Z = np.matmul(weights[keyW], cache[keyA_p]) + weights[keyb]

        if la != L:
            A = (np.exp(Z) - np.exp(-Z)) / (np.exp(Z) + np.exp(-Z))
            D = np.random.rand(A.shape[0], A.shape[1])
            D = np.where(D < keep_prob, 1, 0)
            cache[keyD] = D
            A *= D
            A /= keep_prob
            cache[keyA] = A
        else:
            # Softmax
            t = np.exp(Z)
            A = t / np.sum(t, axis=0, keepdims=True)
            cache[keyA] = A

    return cache
