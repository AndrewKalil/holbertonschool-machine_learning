#!/usr/bin/env python3
""" Biderection Recurrent Neural Network: Forward """

import numpy as np


class BidirectionalCell:
    """
        class BidirectionalCell that represents a bidirectional cell of an RNN
    """

    def __init__(self, i, h, o):
        """
        Constructor
        i is the dimensionality of the data
        h is the dimensionality of the hidden states
        o is the dimensionality of the outputs
        Creates the public instance attributes Whf, Whb, Wy, bhf, bhb, by that
          represent the weights and biases of the cell
            Whf and bhfare for the hidden states in the forward direction
            Whb and bhbare for the hidden states in the backward direction
            Wy and byare for the outputs
        The weights should be initialized using a random normal distribution in
          the order listed above
        The weights will be used on the right side for matrix multiplication
        The biases should be initialized as zeros
        """

        # initializating Weights in order
        self.Whf = np.random.normal(size=(h + i, h))  # size = (25, 15)
        self.Whb = np.random.normal(size=(h + i, h))  # size = (25, 15)
        self.Wy = np.random.normal(size=(h + h, o))  # size = (30, 5)

        # initializating bias in order
        self.bhf = np.zeros(shape=(1, h))
        self.bhb = np.zeros(shape=(1, h))
        self.by = np.zeros(shape=(1, o))

    def forward(self, h_prev, x_t):
        """
        x_t is a numpy.ndarray of shape (m, i) that contains the data input
          for the cell
            m is the batch size for the data
        h_prev is a numpy.ndarray of shape (m, h) containing the previous
          hidden state
        Returns: h_next, the next hidden state
        """

        # https://victorzhou.com/blog/intro-to-rnns/

        x = np.concatenate((h_prev, x_t), axis=1)
        h_t = np.tanh(np.dot(x, self.Whf) + self.bhf)

        return h_t

    def backward(self, h_next, x_t):
        """
        x_t is a numpy.ndarray of shape (m, i) that contains the data
          input for the cell
            m is the batch size for the data
        h_next is a numpy.ndarray of shape (m, h) containing the next
          hidden state
        Returns: h_pev, the previous hidden state
        """

        x = np.concatenate((h_next, x_t), axis=1)
        h_pev = np.tanh(np.dot(x, self.Whb) + self.bhb)

        return h_pev

    def output(self, H):
        """
        public instance method that calculates all outputs for the RNN
        Arguments:
        H is a numpy.ndarray of shape (t, m, 2 * h) that contains the
          concatenated hidden states from both directions, excluding their
          initialized states
            t is the number of time steps
            m is the batch size for the data
            h is the dimensionality of the hidden states
        Returns: Y, the outputs
        """

        t = H.shape[0]

        for i in range(t):
            ŷ = np.dot(H[i], self.Wy) + self.by
            ŷ = np.exp(ŷ) / np.sum(np.exp(ŷ), axis=1, keepdims=True)

            Y = np.array([ŷ]) if i == 0 else np.append(Y, [ŷ], axis=0)

        return Y
