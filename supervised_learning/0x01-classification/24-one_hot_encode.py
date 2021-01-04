#!/usr/bin/env python3
"""One Hot Encode"""
import numpy as np


def one_hot_encode(Y, classes):
    matrix = np.zeros((len(Y), classes))

    if Y is not None:
        matrix[np.arange(len(Y)), Y] = 1
        return matrix.T
    else:
        return None
