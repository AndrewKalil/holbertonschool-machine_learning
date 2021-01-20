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

    cache = dict()
    cache['A0'] = X

    for layer in range(L):
        current_W = weights['W' + str(layer+1)]
        current_b = weights['b' + str(layer+1)]
        A_prev = cache['A' + str(layer)]
        z = (np.matmul(current_W, A_prev)) + current_b
        dropout = np.random.binomial(1, keep_prob, size=z.shape)

        if layer == L:
            t = np.exp(z)
            cache['A' + str(layer+1)] = t / np.sum(t, axis=0, keepdims=True)
        else:
            cache['A' + str(layer+1)] = np.tanh(z)
            cache['D' + str(layer+1)] = dropout
            cache['A' + str(layer+1)] *= dropout
            cache['A' + str(layer+1)] /= keep_prob

    return cache
