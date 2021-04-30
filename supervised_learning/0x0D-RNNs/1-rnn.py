#!/usr/bin/env python3
"""  RNN """
import numpy as np


def rnn(rnn_cell, X, h_0):
    """
    performs forward propagation for a simple RNN
    rnn_cell is an instance of RNNCell that will be used for the forward
      propagation
    X is the data to be used, given as a numpy.ndarray of shape (t, m, i)
        t is the maximum number of time steps
        m is the batch size
        i is the dimensionality of the data
    h_0 is the initial hidden state, given as a numpy.ndarray of shape (m, h)
        h is the dimensionality of the hidden state
    Returns: H, Y
        H is a numpy.ndarray containing all of the hidden states
        Y is a numpy.ndarray containing all of the outputs
    """
    h_prev = h_0
    H = np.array(([h_0]))

    t = X.shape[0]
    for i in range(t):
        h_prev, y = rnn_cell.forward(h_prev, X[i])

        H = np.append(H, [h_prev], axis=0)
        Ŷ = np.array([y]) if i == 0 else np.append(Ŷ, [y], axis=0)

    return H, Ŷ
