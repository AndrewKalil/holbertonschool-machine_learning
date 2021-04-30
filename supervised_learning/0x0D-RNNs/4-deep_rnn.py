#!/usr/bin/env python3
""" Forwar propagation of Deep Recurrent Neral Network """
import numpy as np


def deep_rnn(rnn_cells, X, h_0):
    """
    rnn_cells is a list of RNNCell instances of length l that will be used for
      the forward propagation
        l is the number of layers
    X is the data to be used, given as a numpy.ndarray of shape (t, m, i)
        t is the maximum number of time steps
        m is the batch size
        i is the dimensionality of the data
    h_0 is the initial hidden state, given as a numpy.ndarray of
      shape (l, m, h)
        h is the dimensionality of the hidden state
    Returns: H, Y
        H is a numpy.ndarray containing all of the hidden states
        Y is a numpy.ndarray containing all of the outputs
    """

    h_prev = h_0
    H = np.array(([h_prev]))
    H = np.repeat(H, X.shape[0] + 1, axis=0)

    for i in range(X.shape[0]):
        for a_layer, cell in enumerate(rnn_cells):

            # forwarding
            parameter = X[i] if a_layer == 0 else h_prev
            h_prev, y = cell.forward(H[i, a_layer], parameter)

            # update the hidden states
            H[i + 1, a_layer] = h_prev

            # update all the outputs
            if (i != 0):
                Y[i] = y

            else:
                Y = np.array([y])
                Y = np.repeat(Y, X.shape[0], axis=0)

    return H, Y
