#!/usr/bin/env python3
"""One Hot Encode"""
import numpy as np


def one_hot_encode(Y, classes):
    """ Function for one hot encoding"""

    if Y is None or classes < 1 or type(classes) != int:
        return None

    matrix = np.zeros((len(Y), classes))
    matrix[np.arange(len(Y)), Y] = 1

    return matrix.T
