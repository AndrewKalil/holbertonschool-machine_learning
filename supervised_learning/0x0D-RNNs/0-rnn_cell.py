#!/usr/bin/env python3
""" Recurrent Neural Network Cell """
import numpy as np


class RNNCell():
    """ a cell of a simple RNN: """

    def __init__(self, i, h, o):
        """
        Instantion method
        i is the dimensionality of the data
        h is the dimensionality of the hidden state
        o is the dimensionality of the outputs
        Creates the public instance attributes Wh, Wy, bh, by that represent
          the weights and biases of the cell
            Wh and bh are for the concatenated hidden state and input data
            Wy and by are for the output
        """
        self.Wh = np.random.normal(size=(h + i, h))
        self.Wy = np.random.normal(size=(h, o))
        self.bh = np.zeros(shape=(1, 0))
        self.by = np.zeros(shape=(1, 0))

    def forward(self, h_prev, x_t):
        """
        Forward propagation for a 1 time step
        x_t is a numpy.ndarray of shape (m, i) that contains the data input for
          the cell
            m is the batche size for the data
        h_prev is a numpy.ndarray of shape (m, h) containing the previous
          hidden state
        """
        # hₜ = tanh((Wₓₕ · xₜ) + (Wₕₕ · hₜ₋₁ + bₕ))
        x = np.concatenate((h_prev, x_t), axis=1)
        h_t = np.tanh(np.dot(x, self.Wh) + self.bh)

        ŷ = np.dot(h_t, self.Wy) + self.by
        y = (np.exp(ŷ) / np.sum(np.exp(ŷ), axis=1, keepdims=True))

        return h_t, y
